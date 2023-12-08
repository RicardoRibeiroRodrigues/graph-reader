import cv2
import numpy as np
from utils import *
import os
import json
import pytesseract
import re

DEBUG = os.environ.get("DEBUG", False)


class SynteticGraphPipeline:
    def __init__(self, graph_bounding_box: tuple):
        self.threshhold_config = None
        self.graph_bounding_box = get_bounding_box(graph_bounding_box)

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
        min_x = self.labels_bounding_box[0]
        min_y = self.labels_bounding_box[1]
        max_x = self.labels_bounding_box[2]
        max_y = self.labels_bounding_box[3]

        # Invert the y-coordinate before normalization
        inverted_y = max_y - y

        # Normalize the coordinates
        normalized_x = (x - min_x) / (max_x - min_x)
        normalized_y = (inverted_y - min_y) / (max_y - min_y)
        return normalized_x, normalized_y

    def threshold_image(self, img, low_threshold=50, high_threshold=150, k=5):
        # Convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        _, thresh1 = cv2.threshold(
            gray, low_threshold, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV
        )
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

    def process_detected_text(self, text_list):
        processed_list = []

        for text in text_list:
            # Utilize expressão regular para capturar números decimais
            numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
            # Converta para float se forem números decimais
            numbers = [float(num) for num in numbers]
            for i in range(len(numbers)):
                processed_list.append(numbers[i])

        return processed_list

    def detect_contours(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Performing OTSU threshold
        _, thresh1 = cv2.threshold(
            gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV
        )

        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

        # Applying dilation on the threshold image
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

        # Finding contours
        contours, _ = cv2.findContours(
            dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        return contours

    def process_image(self, image, b_box_graph, b_box_x, b_box_y):
        x_min, y_min, x_max, y_max = get_bounding_box(b_box_graph)
        graph = image[y_min:y_max, x_min:x_max]

        x_min, y_min, x_max, y_max = get_bounding_box(b_box_x)
        axis_x = image[y_min:y_max, x_min:x_max]

        x_min, y_min, x_max, y_max = get_bounding_box(b_box_y)
        axis_y = image[y_min:y_max, x_min:x_max]

        graph_contours = self.detect_contours(graph)
        axis_x_contours, text_x = self.detect_text(axis_x, "x")
        axis_y_contours, text_y = self.detect_text(axis_y, "y")

        processed_text_x = self.process_detected_text(text_x)
        processed_text_y = self.process_detected_text(text_y)

        print("Before uniform")
        print(processed_text_x)
        print(processed_text_y)

        # invert the order of axis_y_contours and axis_x_contours
        axis_x_contours = axis_x_contours[::-1]
        axis_y_contours = axis_y_contours[::-1]

        # Assume uniform spacing between the points in the graph
        # processed_text_x = np.linspace(
        #     processed_text_x[0], processed_text_x[-1], len(axis_x_contours)
        # )
        # processed_text_y = np.linspace(
        #     processed_text_y[0], processed_text_y[-1], len(axis_y_contours)
        # )

        print("After uniform")
        print(processed_text_x)
        print(processed_text_y)

        # Draw contours
        cv2.drawContours(graph, graph_contours, -1, (0, 0, 255), 3)

        i = 0
        j = 0
        min_x = min_y = np.inf
        max_x = max_y = 0
        for contour in axis_x_contours:
            x, y, w, h = cv2.boundingRect(contour)
            bound_middle_x = x + w / 2
            min_x = min(min_x, bound_middle_x)
            max_x = max(max_x, bound_middle_x)

            if i < len(processed_text_x):
                cv2.putText(
                    axis_x,
                    str(processed_text_x[i]),
                    (x + 30, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
                i += 1
            # x_list.append(extract_text(axis_x[y:y+h, x:x+w]))
            cv2.rectangle(axis_x, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for contour in axis_y_contours:
            x, y, w, h = cv2.boundingRect(contour)
            bound_middle_y = y + h / 2
            min_y = min(min_y, bound_middle_y)
            max_y = max(max_y, bound_middle_y)

            if j < len(processed_text_y):
                cv2.putText(
                    axis_y,
                    str(processed_text_y[j]),
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
                j += 1
            cv2.rectangle(axis_y, (x, y), (x + w, y + h), (0, 255, 0), 2)

        debug_image(image)
        self.labels_bounding_box = (min_x, min_y, max_x, max_y)
        self.max_x_value = max(processed_text_x)
        self.min_x_value = min(processed_text_x)
        self.max_y_value = max(processed_text_y)
        self.min_y_value = min(processed_text_y)

    def detect_text(self, img, var):
        lista_dados = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Performing OTSU threshold
        _, thresh1 = cv2.threshold(
            gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV
        )

        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

        # Applying dilation on the threshold image
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

        # Finding contours
        contours, _ = cv2.findContours(
            dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        # Scale img to 300 dpi
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        if var == "x":
            altura_x, largura_x, _ = img.shape
            custom_config = r"--psm 6 -c tessedit_char_whitelist=0123456789.-ABCDEFHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            if altura_x < 30:
                res_x = cv2.resize(
                    img, (largura_x + 20, altura_x + 20), interpolation=cv2.INTER_CUBIC
                )
                debug_image(res_x)
                extracted_text = pytesseract.image_to_string(res_x, config=custom_config)
            else:
                debug_image(img)
                extracted_text = pytesseract.image_to_string(img, config=custom_config)

            lista_dados.append(extracted_text)

        if var == "y":
            custom_config = r"--psm 6 -c tessedit_char_whitelist=0123456789.-"
            altura_y, largura_y, _ = img.shape
            if largura_y < 35:
                res_y = cv2.resize(img, (largura_y + 20, altura_y + 20))
                debug_image(res_y)
                extracted_text = pytesseract.image_to_string(
                    res_y, config=custom_config
                )
            else:
                debug_image(img)
                extracted_text = pytesseract.image_to_string(img, config=custom_config)

            lista_dados.append(extracted_text)

        return contours, lista_dados

    def find_graph_points(self, graph_type: str):
        print(f"Graph type: {graph_type}")
        if graph_type not in ("line", "scatter"):
            return None

        # Find the The thresholded image again, with corrected image.
        # Format x_min, y_min, x_max, y_max
        x_min = self.graph_bounding_box[0]
        y_min = self.graph_bounding_box[1]
        x_max = self.graph_bounding_box[2]
        y_max = self.graph_bounding_box[3]
        thresholded_crop = self.thresholded_image[y_min:y_max, x_min:x_max]

        contours, _ = cv2.findContours(
            thresholded_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            print("Found no contours")
            return None

        mask = np.zeros(self.thresholded_image.shape, np.uint8)
        if graph_type == "line":
            biggest_contour = max(contours, key=cv2.contourArea)
            # Add the padding to the contour, so it is in the original position
            biggest_contour[:, :, 0] += self.graph_bounding_box[0]
            biggest_contour[:, :, 1] += self.graph_bounding_box[1]
            cv2.drawContours(mask, [biggest_contour], 0, 255, -1)
        elif graph_type == "scatter":
            for contour in contours:
                # Add the padding to the contour, so it is in the original position
                contour[:, :, 0] += self.graph_bounding_box[0]
                contour[:, :, 1] += self.graph_bounding_box[1]
                cv2.drawContours(mask, [contour], 0, 255, -1)

        if DEBUG:
            mask_debug = mask.copy()
            if mask.shape > (1000, 1000):
                mask_debug = cv2.resize(mask_debug, (1000, 1000))
            debug_image(mask_debug)

        # Find the graph points as white pixels in the image
        graph_points = np.where(mask == 255)
        # Add the padding to the points, so they are in the original position
        x_coords = graph_points[1]
        y_coords = graph_points[0]
        # Sort both coordinates by x
        sorted_x_coords, sorted_y_coords = zip(
            *sorted(zip(x_coords, y_coords), key=lambda point: point[0])
        )

        unique_points = []
        for i in range(0, len(sorted_x_coords), 2):
            unique_points.append(Point(sorted_x_coords[i], sorted_y_coords[i]))

        str_points = []
        # Calculate the multipliers for the normalized coordinates to the graph labels interval
        x_multiplier = self.max_x_value - self.min_x_value
        y_multiplier = self.max_y_value - self.min_y_value

        for point in unique_points:
            x, y = self.normalize_point(point.x, point.y)
            # Translate the normalized coordinates to the graph labels interval
            x = self.min_x_value + x * x_multiplier
            y = self.min_y_value + y * y_multiplier
            str_points.append(f"{x},{y}\n")

        csv_str = "".join(str_points)
        # DEBUG
        if DEBUG:
            with open("out.csv", "w") as f:
                f.write(csv_str)

        return csv_str.encode("utf-8")
