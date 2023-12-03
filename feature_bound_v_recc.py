from flask import Flask, request, jsonify, render_template, make_response
from utils import *
import cv2
import numpy as np
import io
import json
import pytesseract
from collections import defaultdict
import re
import matplotlib.pyplot as plt
import matplotlib
import tkinter
matplotlib.use('TkAgg')
#plt.switch_backend('TkAgg')

app = Flask(__name__)


def detect_text(img, var):
    lista_dados = []
    graph_points = None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    # Performing OTSU threshold
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    
    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

    # Finding contours
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Redução de ruído e aumento de contraste
 
    thr = cv2.adaptiveThreshold(gray, 255, 
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 23)

    if var == 'x':
        altura_x, largura_x, canais_x = img.shape
        print(altura_x)
        if altura_x < 30:
            res_x = cv2.resize(img, (largura_x+20, altura_x+20), interpolation=cv2.INTER_CUBIC)
            extracted_text = pytesseract.image_to_string(res_x, config='--psm 6')
        else: 
            extracted_text = pytesseract.image_to_string(img, config='--psm 6')

      
        lista_dados.append(extracted_text)
    
    if var == 'y':
        custom_config = r"--psm 6 -c tessedit_char_whitelist=0123456789.-"
        altura_y, largura_y, canais_y = img.shape
        if largura_y < 35:
            res_y = cv2.resize(img, (largura_y+20, altura_y+20))
            extracted_text = pytesseract.image_to_string(res_y, config=custom_config)
        else:
            extracted_text = pytesseract.image_to_string(img, config=custom_config)

        lista_dados.append(extracted_text)

    if var=="grafico":
        # Dicionário para armazenar os pontos do gráfico por coordenada x
        graph_points = defaultdict(list)

        # Percorre a imagem para encontrar os pontos não brancos
        for y in range(thr.shape[0]):
            for x in range(thr.shape[1]):
                pixel_value = thr[y, x]
                if pixel_value < 255:  # Se o pixel não for branco
                    graph_points[x].append(-y)  # Armazena o valor y para a coordenada x

        # Calcula a média dos valores de y para cada coordenada x
        for x, y_values in graph_points.items():
            graph_points[x] = float(np.mean(y_values))
 
    return contours, lista_dados, graph_points

def get_bounding_box(b_box_json):
    x_min = round(b_box_json['x_min'])
    y_min = round(b_box_json['y_min'])
    x_max = x_min + round(b_box_json['width'])
    y_max = y_min + round(b_box_json['height'])
    return x_min, y_min, x_max, y_max

def corrigir_intervalo(lista):
    nova_lista = []
    
    if len(lista) < 2:
        return lista  # Retorna a lista original se tiver menos de dois elementos

    intervalo = lista[3] - lista[2] # [0 1 2 3 4]

    # Adiciona o primeiro número à nova lista
    nova_lista.append(lista[3] - 3*intervalo)

    # Continua a sequência com base no intervalo detectado
    for i in range(1, len(lista)):
        # Verifica se o número é um float
        if isinstance(lista[i], float):
            # Calcula o intervalo considerando a parte fracionária
            diff = lista[i] - lista[i - 1]
            decimal_diff = round(diff % 1, 1)  # Obtém a parte fracionária e arredonda para 1 casa decimal

            if decimal_diff != intervalo:
                # Se a diferença não estiver correta, calcula o novo número
                nova_lista.append(round(nova_lista[-1] + intervalo, 1))
            else:
                # Mantém o número original na lista
                nova_lista.append(lista[i])
        elif isinstance(lista[i], int):
            # Para números inteiros, aplica a lógica anteriormente utilizada
            if lista[i] - lista[i - 1] != intervalo:
                nova_lista.append(nova_lista[-1] + intervalo)
            else:
                nova_lista.append(lista[i])
        else:
            # Mantém qualquer outro tipo de dado na lista
            nova_lista.append(lista[i])

    return nova_lista

def process_detected_text(text_list):
    processed_list = []

    for text in text_list:
        # Utilize expressão regular para capturar números decimais
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
        numbers = [float(num) for num in numbers]  # Converta para float se forem números decimais
        
        for i in range(len(numbers)):
            processed_list.append(numbers[i])

    return processed_list

def process_eixoX_text(text_list):
    processed_list = []

    for text in text_list:
        # Remover \n e dividir os itens conforme os espaços em branco
        text_without_newline = text.replace('\n', '')
        processed_items = text_without_newline.split()
        processed_list.extend(processed_items)

    return processed_list


def process_image(image_data, b_box_graph, b_box_x, b_box_y, width, height):
    # Decode the base64 image data and convert it to a NumPy array
    image_bytes = io.BytesIO(image_data)
    image_array = np.frombuffer(image_bytes.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract bounding box coordinates from JSON string
    box_graph = json.loads(b_box_graph)
    box_axis_x = json.loads(b_box_x)
    box_axis_y = json.loads(b_box_y)

    x_min_graf, y_min_graf, x_max_graf, y_max_graf = get_bounding_box(box_graph)
    graph = image[y_min_graf:y_max_graf, x_min_graf:x_max_graf]

    x_min_box_x, y_min_box_x, x_max_box_x, y_max_box_x = get_bounding_box(box_axis_x)
    axis_x = image[y_min_box_x:y_max_box_x, x_min_box_x:x_max_box_x]

    x_min_box_y, y_min_box_y, x_max_box_y, y_max_box_y = get_bounding_box(box_axis_y)
    axis_y = image[y_min_box_y:y_max_box_y, x_min_box_y:x_max_box_y]

    graph_contours, texto_grafico, pontos_graf = detect_text(graph, "grafico")
    axis_x_contours, texto_x, xxx = detect_text(axis_x, "x")
    axis_y_contours, texto_y, yyy = detect_text(axis_y, "y")

    # Draw contours
    cv2.drawContours(graph, graph_contours, -1, (0, 0, 255), 3)

    for contour in axis_x_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(axis_x, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    for contour in axis_y_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(axis_y, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Processamento dos dados do eixo X e eixo Y
    processed_text_x = process_detected_text(texto_x)
    processed_text_y = process_detected_text(texto_y)
    filtro_y = corrigir_intervalo(processed_text_y)
    filtro_x = corrigir_intervalo(processed_text_x)
    eixo_x_texto = process_eixoX_text(texto_x)


    print("Eixo X não-processado:", texto_x)
    print("Eixo X TEXTO processado:", eixo_x_texto)
    print("Eixo X processado:", processed_text_x)
    print("Eixo X - Corrigido (se necessário)", filtro_x)
    print("------------------------")
    print("Eixo Y não-processado", texto_y)
    print("Eixo Y processado:", processed_text_y)
    print("Eixo Y - Corrigido (se necessário)", filtro_y)
    print("------------------------")
    #print("Esses são os pontos do gráfico:", pontos_graf)
    print("------------------------")
    print("------------------------")

    scale_y = (max(filtro_y) - min(filtro_y)) / (y_max_box_y - y_min_box_y)
    real_values_y = [(-pixel_value + y_min_box_y) * (-scale_y) + max(filtro_y) for pixel_value in pontos_graf.values()]

    # Escalas de conversão entre pixels e valores reais nos eixos X e Y
    if isinstance(filtro_x, float):
        scale_x = (max(filtro_x) - min(filtro_x)) / (x_max_box_x - x_min_box_x)
        real_values_x = [(-pixel_value + x_min_box_x) * (-scale_x) + max(filtro_x) for pixel_value in range(x_min_box_x, x_max_box_x)]
        plot_graph(pontos_graf, None, real_values_y, real_values_x)
    else:
        print("Tentativa de conversão de escala:" , real_values_y)
        plot_graph(pontos_graf, eixo_x_texto, real_values_y, None)

    # Return the processed image as a base64 encoded string
    _, img_encoded = cv2.imencode('.png', image)
    img_base64 = img_encoded.tobytes()

    return img_base64, pontos_graf

def process_handdrawn_graph(image_data):
    image_bytes = io.BytesIO(image_data)
    image_array = np.frombuffer(image_bytes.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    # Performing OTSU threshold
    _, thresh1 = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
    
    # Finding contours
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, (0, 0, 255), 3)

    graph_b_box = find_handdrawn_bbox(image)
    graph_data = get_graph_data_norm(image, graph_b_box)

    # Save as csv
    with open('graph_data.csv', 'w') as f:
        f.write('x,y\n')
        for point in graph_data:
            f.write(f'{point[0]},{point[1]}\n')

    # Return the processed image as a base64 encoded string
    _, img_encoded = cv2.imencode('.png', image)
    img_base64 = img_encoded.tobytes()
    return img_base64 


    # Verifica se os valores do eixo Y são todos números
    y_values = list(axis_values.values())

    if all(isinstance(value, (int, float)) for value in y_values):
        min_y = min(y_values)
        max_y = max(y_values)

        # Normaliza os valores do eixo Y no intervalo [0, 1]
        normalized_y_values = [(val - min_y) / (max_y - min_y) for val in y_values]

        # Atualiza os valores do dicionário com os valores normalizados do eixo Y
        for i, (key, _) in enumerate(axis_values.items()):
            axis_values[key] = normalized_y_values[i]

    # Retorna o dicionário modificado ou não modificado se não forem apenas números
    return axis_values


def plot_graph(graph_points, eixo_x_texto, eixo_y, eixo_x):
    # Ordena os pontos pela chave (coordenada x)
    sorted_points = graph_points.items()

    # Separa as coordenadas x e y ordenadas
    x_values, y_values = zip(*sorted_points)

    # Filtro para ver se eixo X é texto ou não.
    if eixo_x_texto is None:
        eixoX = eixo_x
    else:
        eixoX = eixo_x_texto

    # Ajuste do eixo Y para ter o mesmo número de pontos que o eixo X
    if len(eixo_y) > len(eixoX):
        eixo_y = eixo_y[:len(eixoX)]  # Reduz o eixo Y para o mesmo comprimento do eixo X
    elif len(eixo_y) < len(eixoX):
        eixoX = np.linspace(min(eixoX), max(eixoX), len(eixo_y))

    print("Esse é o eixo_X final: ", eixoX)
    print("Esse é o eixo_y final: ", eixo_y)

    # Plota o gráfico
    plt.figure(figsize=(8, 6))
    plt.plot(eixoX, eixo_y, linestyle='-')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.title('Gráfico a partir dos pontos detectados')
    plt.grid(True)
    plt.show()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-image', methods=['POST'])
def process_image_route():
    try:
        is_synthetic = True if request.form['isSynthetic'] == "true" else False
        image_data = request.files['image'].read()
        width = request.form['width']
        height = request.form['height']
        if is_synthetic:
            graph_box = request.form['graphBox']
            axis_x_box = request.form['axisXBox']
            axis_y_box = request.form['axisYBox']
            processed_image_data = process_image(image_data, graph_box, axis_x_box, axis_y_box, width, height)
        else:
            processed_image_data = process_handdrawn_graph(image_data)
        response = make_response(processed_image_data)
        response.headers['Content-Type'] = 'image/png'
        return response
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
