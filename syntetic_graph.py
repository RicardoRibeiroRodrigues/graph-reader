import cv2
import numpy as np
from utils import *
import os
import json
import pytesseract
from collections import defaultdict

DEBUG = os.environ.get("DEBUG", False)


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __hash__(self):
        """
        Hash the point using the x coordinate to allow the x coordinate to be unique
        """
        return hash(self.x)


class SynteticGraphPipeline:
    def __init__(self):
        self.found_bbox = None
        self.found_bbox_padded = None
        self.lines_intersec = None
        self.threshhold_config = None

    def toDict(self):
        return {
            "found_bbox": self.found_bbox,
            "found_bbox_padded": self.found_bbox_padded,
            "lines_intersec": self.lines_intersec,
            "threshhold_config": self.threshhold_config,
        }

    @classmethod
    def fromDict(cls, d):
        pipeline = cls()
        pipeline.found_bbox = d["found_bbox"]
        pipeline.found_bbox_padded = d["found_bbox_padded"]
        pipeline.lines_intersec = d["lines_intersec"]
        pipeline.threshhold_config = d["threshhold_config"]
        return pipeline

    def normalize_point(self, shape, x, y):
        min_y = self.found_bbox[1]
        min_x = self.found_bbox[0]

        # Invert the y-coordinate before normalization
        inverted_y = shape[0] - y

        # Normalize the coordinates
        normalized_x = (x - min_x) / (self.found_bbox[2] - min_x)
        normalized_y = (inverted_y - min_y) / (self.found_bbox[3] - min_y)
        return normalized_x, normalized_y

    def threshold_image(self, img, low_threshold=50, high_threshold=150, blur_amount=3, k=5):
        # Convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray = cv2.equalizeHist(gray)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # Optionally apply dilation based on the characteristics of synthetic plots
        if k > 1:
            kernel = np.ones((k, k), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

        self.threshhold_config = {
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "k": k,
        }

        return cv2_image_to_bytes(edges)

    def get_bounding_box(b_box_json):
        x_min = round(b_box_json['x_min'])
        y_min = round(b_box_json['y_min'])
        x_max = x_min + round(b_box_json['width'])
        y_max = y_min + round(b_box_json['height'])
        return x_min, y_min, x_max, y_max

    def detect_text(img, var):
        lista_dados = []
        graph_points = None
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

        if var == 'x':
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

        if var == 'y':
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

        if var == "grafico":
            # Dicionário para armazenar os pontos do gráfico por coordenada x
            graph_points = defaultdict(list)

            # Percorre a imagem para encontrar os pontos não brancos
            for y in range(thr.shape[0]):
                for x in range(thr.shape[1]):
                    pixel_value = thr[y, x]
                    if pixel_value < 255:  # Se o pixel não for branco
                        # Armazena o valor y para a coordenada x
                        graph_points[x].append(-y)

            # Calcula a média dos valores de y para cada coordenada x
            for x, y_values in graph_points.items():
                graph_points[x] = float(np.mean(y_values))

        return contours, lista_dados, graph_points
