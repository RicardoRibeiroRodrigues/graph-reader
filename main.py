from flask import Flask, request, jsonify, render_template, make_response
from utils import *
import cv2
import numpy as np
import json
import pytesseract
from hand_graph import HandDrawnGraphPipeline

handdrawn_graph_pipeline = None

app = Flask(__name__)

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

def detect_text(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    # Performing OTSU threshold
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    
    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

    # Finding contours
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    return contours

def get_bounding_box(b_box_json):
    x_min = round(b_box_json['x_min'])
    y_min = round(b_box_json['y_min'])
    x_max = x_min + round(b_box_json['width'])
    y_max = y_min + round(b_box_json['height'])
    return x_min, y_min, x_max, y_max


def process_image(image_data, b_box_graph, b_box_x, b_box_y, width, height):
    image = cv2_image_from_bytes(image_data)
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
    

    graph_contours = detect_text(graph)
    axis_x_contours = detect_text(axis_x)
    axis_y_contours = detect_text(axis_y)

    # Draw contours
    cv2.drawContours(graph, graph_contours, -1, (0, 0, 255), 3)

    x_list = []
    for contour in axis_x_contours:
        x, y, w, h = cv2.boundingRect(contour)
        # x_list.append(extract_text(axis_x[y:y+h, x:x+w]))
        cv2.rectangle(axis_x, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    y_list = []
    for contour in axis_y_contours:
        x, y, w, h = cv2.boundingRect(contour)
        y_list.append(extract_text(axis_y[y:y+h, x:x+w]))
        cv2.rectangle(axis_y, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print(x_list)
    print(y_list)
    # Return the processed image as a base64 encoded string
    return cv2_image_to_bytes(image)


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
        "blur": 3
    }
    return render_template('handdrawn.html', **default_configs)

@app.route('/process-image-syntetic', methods=['POST'])
def process_syntetic_graph():
    try:
        image_data = request.files['image'].read()
        width = request.form['width']
        height = request.form['height']
        graph_box = request.form['graphBox']
        axis_x_box = request.form['axisXBox']
        axis_y_box = request.form['axisYBox']
        processed_image_data = process_image(image_data, graph_box, axis_x_box, axis_y_box, width, height)
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

        global handdrawn_graph_pipeline
        handdrawn_graph_pipeline = HandDrawnGraphPipeline(image)
        
        processed_image_data = handdrawn_graph_pipeline.threshold_image(threshold, blur, k)
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
        print(request.form)

        if 'line_threshold' in request.form:
            line_theshold = int(request.form['line_threshold'])
        if 'min_line_percent' in request.form:
            min_line_percent = float(request.form['min_line_percent'])
        if 'max_line_gap' in request.form:
            max_line_gap = int(request.form['max_line_gap'])
        
        print(line_theshold, min_line_percent, max_line_gap)
        img = handdrawn_graph_pipeline.find_lines(line_theshold, min_line_percent, max_line_gap)
        response = make_response(img)
        response.headers['Content-Type'] = 'image/png'
        return response
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})

@app.route('/process-image-hand-2', methods=['POST'])
def process_hand_graph_2():
    try:
        pad_size_y = pad_size_x = 30
        if 'pad_size_x' in request.form:
            pad_size_x = int(request.form['pad_size_x'])
        if 'pad_size_y' in request.form:
            pad_size_y = int(request.form['pad_size_y'])
        
        res = handdrawn_graph_pipeline.find_handdrawn_bbox(pad_size_x, pad_size_y)
        response = make_response(res)
        response.headers['Content-Type'] = 'image/png'
        return response
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})

@app.route('/process-image-hand-3', methods=['POST'])
def process_hand_graph_3():
    try:
        graph_type = 'line'
        img = handdrawn_graph_pipeline.find_graph_points(graph_type)
        response = make_response(img)
        response.headers['Content-Type'] = 'image/png'
        return response
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
