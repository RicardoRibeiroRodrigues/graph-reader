from flask import Flask, request, jsonify, render_template, make_response, session
from utils import *
import cv2
import numpy as np
import json
import pytesseract
from hand_graph import HandDrawnGraphPipeline
from syntetic_graph import SynteticGraphPipeline
import os
from dotenv import load_dotenv

load_dotenv()
DEBUG = os.environ.get('DEBUG', False)
if not DEBUG:
    # Find the path to the tesseract executable
    pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_PATH', "./tesseract")



app = Flask(__name__)
app.secret_key = os.urandom(24)


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
        "pad_size_x": .1,
        "pad_size_y": .1,
    }
    return render_template('handdrawn.html', **default_configs)


@app.route('/process-image-syntetic', methods=['POST'])
def process_syntetic_graph():
    try:
        image_data = request.files['image'].read()
        graph_box = request.form['graphBox']
        axis_x_box = request.form['axisXBox']
        axis_y_box = request.form['axisYBox']
        is_axis_x_rotated = request.form['isXAxisRotated']
        is_axis_x_rotated = is_axis_x_rotated == 'true'
        print(f"{is_axis_x_rotated=}")
        image = cv2_image_from_bytes(image_data)
        
        graph_box = json.loads(graph_box)
        axis_x_box = json.loads(axis_x_box)
        axis_y_box = json.loads(axis_y_box)

        syntetic_graph_pipeline = SynteticGraphPipeline(graph_box)
        syntetic_graph_pipeline.process_image(
            image.copy(), graph_box, axis_x_box, axis_y_box, is_axis_x_rotated
        )

        syntetic_graph_pipeline.threshold_image(
            image, low_threshold=20, high_threshold=150, k=5
        )

        plot_type = request.form['plotType']
        # Remove the "" from the string
        plot_type = plot_type[1:-1]

        # Processamento adicional: encontrar pontos do gráfico
        csv_data = syntetic_graph_pipeline.find_graph_points(plot_type)
        session['pipeline'] = syntetic_graph_pipeline.toDict()

        # Retorna o CSV com as coordenadas dos pontos do gráfico
        response = make_response(csv_data)
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = 'attachment; filename=graph.csv'
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
        pad_size_y = pad_size_x = 0.1
        if 'pad_size_x' in request.form:
            pad_size_x = float(request.form['pad_size_x'])
        if 'pad_size_y' in request.form:
            pad_size_y = float(request.form['pad_size_y'])
        
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
