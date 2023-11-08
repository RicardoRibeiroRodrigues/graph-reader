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

def extract_text(img) -> str:
    img = img.copy()
    # dilate and erode
    _, img = cv2.threshold(img, 0, 200, cv2.THRESH_BINARY)
    debug_image(img)
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
    app.run(debug=True)
