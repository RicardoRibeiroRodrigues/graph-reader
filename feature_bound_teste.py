from flask import Flask, request, jsonify, render_template, make_response, Response
import cv2
import numpy as np
import io
import json
import pytesseract

app = Flask(__name__)


def debug_image(img):
    cv2.imshow('Image', img)
    while True:
        if cv2.waitKey(0) == ord('q'):
            break
    cv2.destroyAllWindows()


def detect_text(img, var):
    lista_dados = []
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
    
    #thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    if var == 'x':
        extracted_text = pytesseract.image_to_string(thr, config='--psm 6')
        lista_dados.append(extracted_text)
    
    if var == 'y':
        extracted_text = pytesseract.image_to_string(thr, config='--psm 6')
        lista_dados.append(extracted_text)
    

    return contours, lista_dados

def get_bounding_box(b_box_json):
    x_min = round(b_box_json['x_min'])
    y_min = round(b_box_json['y_min'])
    x_max = x_min + round(b_box_json['width'])
    y_max = y_min + round(b_box_json['height'])
    return x_min, y_min, x_max, y_max

def process_detected_text(text_list):
    processed_list = []
    last_valid_number = None

    for text in text_list:
        # Remove quebras de linha e espaços em branco e separa os números
        numbers = [int(num.strip()) for num in text.split() if num.strip().replace('-', '').isdigit()]
        
        if numbers:
            # Verifica se há uma diferença maior que um intervalo entre números consecutivos
            if last_valid_number is not None and numbers[0] - last_valid_number > 1:
                # Corrige o número seguindo a lógica da escala
                numbers[0] = last_valid_number + 1
            
            last_valid_number = numbers[-1]
            processed_list.extend(numbers)
    
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

    x_min, y_min, x_max, y_max = get_bounding_box(box_graph)
    graph = image[y_min:y_max, x_min:x_max]

    x_min, y_min, x_max, y_max = get_bounding_box(box_axis_x)
    axis_x = image[y_min:y_max, x_min:x_max]

    x_min, y_min, x_max, y_max = get_bounding_box(box_axis_y)
    axis_y = image[y_min:y_max, x_min:x_max]
    

    graph_contours, texto_grafico = detect_text(graph, "qualquer")
    axis_x_contours, texto_x = detect_text(axis_x, "x")
    axis_y_contours, texto_y = detect_text(axis_y, "y")

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

    print("Eixo X não-processado:", texto_x)
    print("Eixo X processado:", processed_text_x)
    print("------------------------")
    print("Eixo Y não-processado", texto_y)
    print("Eixo Y processado:", processed_text_y)
    

    # Return the processed image as a base64 encoded string
    _, img_encoded = cv2.imencode('.png', image)
    img_base64 = img_encoded.tobytes()
    return img_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-image', methods=['POST'])
def process_image_route():
    try:
        image_data = request.files['image'].read()
        graph_box = request.form['graphBox']
        axis_x_box = request.form['axisXBox']
        axis_y_box = request.form['axisYBox']
        width = request.form['width']
        height = request.form['height']

        processed_image_data = process_image(image_data, graph_box, axis_x_box, axis_y_box, width, height)
        response = make_response(processed_image_data)
        response.headers['Content-Type'] = 'image/png'
        return response

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
