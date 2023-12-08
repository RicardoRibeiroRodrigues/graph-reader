import cv2
import numpy as np
from utils import *
import os
import json
import pytesseract
from collections import defaultdict

DEBUG = os.environ.get("DEBUG", False)


class SynteticGraphPipeline:
    def __init__(self, graph_bounding_box: tuple, min_x_value: float, max_x_value: float, min_y_value: float, max_y_value: float):
        self.threshhold_config = None
        self.graph_bounding_box = graph_bounding_box
        # Graph labels interval
        self.min_x_value = min_x_value
        self.max_x_value = max_x_value
        self.min_y_value = min_y_value
        self.max_y_value = max_y_value


    def toDict(self):
        return {
            "threshhold_config": self.threshhold_config,
            "graph_bounding_box": self.graph_bounding_box,
        }

    @classmethod
    def fromDict(cls, d):
        pipeline = cls()
        pipeline.threshhold_config = d["threshhold_config"]
        pipeline.graph_bounding_box = d["graph_bounding_box"]
        return pipeline

    def normalize_point(self, x, y):
        # TODO: Esse aqui tem que ser os pontos adquiridos a partir dos labels
        min_y = self.graph_bounding_box[1]
        min_x = self.graph_bounding_box[0]
        max_x = self.graph_bounding_box[2]
        max_y = self.graph_bounding_box[3]

        # Invert the y-coordinate before normalization
        inverted_y = max_y - y

        # Normalize the coordinates
        normalized_x = (x - min_x) / (max_x - min_x)
        normalized_y = (inverted_y - min_y) / (max_y - min_y)
        return normalized_x, normalized_y

    def threshold_image(self, img, low_threshold=50, high_threshold=150, blur_amount=3, k=5):
        # Convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        edges = thresh1

        # Optionally apply dilation based on the characteristics of synthetic plots
        if k > 1:
            kernel = np.ones((k, k), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            edges = cv2.erode(edges, kernel, iterations=1)
        

        self.threshhold_config = {
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "k": k,
        }
        self.thresholded_image = edges
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
            thresholded_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
