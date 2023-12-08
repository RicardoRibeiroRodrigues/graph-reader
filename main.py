from flask import Flask, request, jsonify, render_template, make_response, session
from utils import *
import cv2
import numpy as np
import json
import pytesseract
from hand_graph import HandDrawnGraphPipeline
import os
import re

DEBUG = os.environ.get('DEBUG', False)
if not DEBUG:
    # Find the path to the tesseract executable
    pytesseract.pytesseract.tesseract_cmd = os.environ.get(
        'TESSERACT_PATH', "/usr/bin/tesseract")


app = Flask(__name__)
app.secret_key = os.urandom(24)


def extract_text(img) -> str:
    img = img.copy()
    # dilate and erode
    _, img = cv2.threshold(img, 0, 200, cv2.THRESH_BINARY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # PSM 8: Treat the image as a single word
    text = pytesseract.image_to_string(img, config='--psm 8')
    return text


def detect_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Performing OTSU threshold
    _, thresh1 = cv2.threshold(
        gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

    # Finding contours
    contours, _ = cv2.findContours(
        dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours


def get_bounding_box(b_box_json):
    x_min = round(b_box_json['x_min'])
    y_min = round(b_box_json['y_min'])
    x_max = x_min + round(b_box_json['width'])
    y_max = y_min + round(b_box_json['height'])
    return x_min, y_min, x_max, y_max


def process_image(image_data, b_box_graph, b_box_x, b_box_y, width, height):
    image = cv2_image_from_bytes(image_data)

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

    graph_contours = detect_contours(graph)
    axis_x_contours, text_x = detect_text(axis_x, 'x')
    axis_y_contours, text_y = detect_text(axis_y, 'y')

    processed_text_x = process_detected_text(text_x)
    processed_text_y = process_detected_text(text_y)

    print(processed_text_x)
    print(processed_text_y)

    # invert the order of axis_y_contours and axis_x_contours
    axis_x_contours = axis_x_contours[::-1]
    axis_y_contours = axis_y_contours[::-1]

    # Assume uniform spacing between the points in the graph
    processed_text_x = np.linspace(
        processed_text_x[0], processed_text_x[-1], len(axis_x_contours))
    processed_text_y = np.linspace(
        processed_text_y[0], processed_text_y[-1], len(axis_y_contours))

    th_graph = threshold_image(graph)
    # Draw contours
    cv2.drawContours(graph, graph_contours, -1, (0, 0, 255), 3)

    x_list = []
    i = 0
    j = 0

    for contour in axis_x_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if i < len(processed_text_x):
            cv2.putText(axis_x, str(processed_text_x[i]), (x+30, y+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            i += 1
        # x_list.append(extract_text(axis_x[y:y+h, x:x+w]))
        cv2.rectangle(axis_x, (x, y), (x + w, y + h), (0, 255, 0), 2)

    y_list = []
    for contour in axis_y_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if j < len(processed_text_y):
            cv2.putText(axis_y, str(processed_text_y[j]), (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            j += 1
        y_list.append(extract_text(axis_y[y:y+h, x:x+w]))
        cv2.rectangle(axis_y, (x, y), (x + w, y + h), (0, 255, 0), 2)

    white_points = np.column_stack(np.where(th_graph == 255)[::-1])
    # normalize the curve with the points
    normalized_points = normalize_point(white_points, graph)

    # Draw the curve
    highlight_img = highlight_graph(image, normalized_points)


    # Return the processed image as a base64 encoded string
    return cv2_image_to_bytes(highlight_img)


def process_detected_text(text_list):
    processed_list = []

    for text in text_list:
        # Utilize expressão regular para capturar números decimais
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
        # Converta para float se forem números decimais
        numbers = [float(num) for num in numbers]

        for i in range(len(numbers)):
            processed_list.append(numbers[i])

    return processed_list


def detect_text(img, type):
    lista_dados = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Performing OTSU threshold
    _, thresh1 = cv2.threshold(
        gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

    # Finding contours
    contours, _ = cv2.findContours(
        dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Redução de ruído e aumento de contraste

    thr = cv2.adaptiveThreshold(gray, 255,
                                cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 23)

    if type == 'x':
        altura_x, largura_x, canais_x = img.shape
        print(altura_x)
        if altura_x < 30:
            res_x = cv2.resize(img, (largura_x+20, altura_x+20),
                               interpolation=cv2.INTER_CUBIC)
            extracted_text = pytesseract.image_to_string(
                res_x, config='--psm 6')
        else:
            extracted_text = pytesseract.image_to_string(img, config='--psm 6')

        lista_dados.append(extracted_text)

    if type == 'y':
        custom_config = r"--psm 6 -c tessedit_char_whitelist=0123456789.-"
        altura_y, largura_y, canais_y = img.shape
        if largura_y < 35:
            res_y = cv2.resize(img, (largura_y+20, altura_y+20))
            extracted_text = pytesseract.image_to_string(
                res_y, config=custom_config)
        else:
            extracted_text = pytesseract.image_to_string(
                img, config=custom_config)

        lista_dados.append(extracted_text)

    return contours, lista_dados

def draw_line(image, points, color, thickness=2):
    for i in range(len(points) - 1):
        pt1 = (int(points[i][1]), int(points[i][0]))
        pt2 = (int(points[i + 1][1]), int(points[i + 1][0]))
        cv2.line(image, pt1, pt2, color, thickness)


def highlight_graph(image, curve_points):
    highlighted_image = image.copy()

    draw_line(highlighted_image, curve_points, (0, 0, 255), thickness=2)

    return highlighted_image

def normalize_point(points, image):
    normalized_points = points.astype(float)
    normalized_points /= image.shape[1]

    return normalized_points

def threshold_image(img, low_threshold=50, high_threshold=150, blur_amount=3, k=5):
    # Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    # Optionally apply dilation based on the characteristics of synthetic plots
    if k > 1:
        kernel = np.ones((k, k), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

    return cv2_image_to_bytes(edges)


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/syntetic-graph")
def syntetic_graph():
    return render_template('syntetic.html')


@app.route("/handdrawn-graph")
def handdrawn_graph():
    default_configs = {
        "line_threshold": 60,
        "min_line_percentage": 0.25,
        "max_line_gap": 3,
        "threshold": 100,
        "k": 5,
        "blur": 3,
        "pad_size_x": 15,
        "pad_size_y": 15,
    }
    return render_template('handdrawn.html', **default_configs)


@app.route('/process-image-syntetic', methods=['POST'])
def process_syntetic_graph():
    try:
        image_data = request.files['image'].read()
        image = cv2_image_from_bytes(image_data)
        width = request.form['width']
        height = request.form['height']
        graph_box = request.form['graphBox']
        axis_x_box = request.form['axisXBox']
        axis_y_box = request.form['axisYBox']
        processed_image_data = process_image(
            image_data, graph_box, axis_x_box, axis_y_box, width, height)
        response = make_response(processed_image_data)
        response.headers['Content-Type'] = 'image/png'
        return response
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})


@app.route('/process-image-hand', methods=['POST'])
def process_hand_graph():
    try:
        image_data = request.files['image'].read()
        image = cv2_image_from_bytes(image_data)
        threshold = 100
        k = 5
        blur = 3
        if 'threshold' in request.form:
            threshold = int(request.form['threshold'])
        if 'k' in request.form:
            k = int(request.form['k'])
        if 'blur' in request.form:
            blur = int(request.form['blur'])

        handdrawn_graph_pipeline = HandDrawnGraphPipeline()
        processed_image_data = handdrawn_graph_pipeline.threshold_image(
            image, threshold, blur, k)
        session['pipeline'] = handdrawn_graph_pipeline.toDict()

        response = make_response(processed_image_data)
        response.headers['Content-Type'] = 'image/png'
        return response
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})


@app.route('/process-image-hand-1', methods=['POST'])
def process_hand_graph_1():
    try:
        line_theshold = 60
        min_line_percent = 0.25
        max_line_gap = 3
        img = request.files['image'].read()
        img = cv2_image_from_bytes(img)
        th_img = request.files['th_image'].read()
        th_img = cv2_image_from_bytes(th_img, cv2.IMREAD_GRAYSCALE)

        if 'line_threshold' in request.form:
            line_theshold = int(request.form['line_threshold'])
        if 'min_line_percent' in request.form:
            min_line_percent = float(request.form['min_line_percent'])
        if 'max_line_gap' in request.form:
            max_line_gap = int(request.form['max_line_gap'])

        handdrawn_graph_pipeline = HandDrawnGraphPipeline.fromDict(
            session['pipeline'])
        img = handdrawn_graph_pipeline.find_lines(
            th_img, img, line_theshold, min_line_percent, max_line_gap)
        session['pipeline'] = handdrawn_graph_pipeline.toDict()
        response = make_response(img)
        response.headers['Content-Type'] = 'image/png'
        return response
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})


@app.route('/process-image-hand-2', methods=['POST'])
def process_hand_graph_2():
    try:
        pad_size_y = pad_size_x = 15
        if 'pad_size_x' in request.form:
            pad_size_x = int(request.form['pad_size_x'])
        if 'pad_size_y' in request.form:
            pad_size_y = int(request.form['pad_size_y'])

        img = request.files['image'].read()
        img = cv2_image_from_bytes(img)
        th_img = request.files['th_image'].read()
        th_img = cv2_image_from_bytes(th_img, cv2.IMREAD_GRAYSCALE)

        handdrawn_graph_pipeline = HandDrawnGraphPipeline.fromDict(
            session['pipeline'])
        res = handdrawn_graph_pipeline.find_handdrawn_bbox(
            img, th_img, pad_size_x, pad_size_y)
        session['pipeline'] = handdrawn_graph_pipeline.toDict()

        response = make_response(res)
        response.headers['Content-Type'] = 'image/png'
        return response
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})


@app.route('/process-image-hand-3', methods=['POST'])
def process_hand_graph_3():
    try:
        img = request.files['image'].read()
        img = cv2_image_from_bytes(img)
        plot_type = request.form['plotType']
        # Remove the "" from the string
        plot_type = plot_type[1:-1]

        handdrawn_graph_pipeline = HandDrawnGraphPipeline.fromDict(
            session['pipeline'])
        csv = handdrawn_graph_pipeline.find_graph_points(img, plot_type)
        session['pipeline'] = handdrawn_graph_pipeline.toDict()

        response = make_response(csv)
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = 'attachment; filename=graph.csv'
        return response

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
