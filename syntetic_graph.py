import cv2
import numpy as np
from utils import *
import os
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

        # Normalize the coordinates
        normalized_x = (x - min_x) / (max_x - min_x)
        normalized_y = (y - min_y) / (max_y - min_y)
        normalized_y = 1 - normalized_y
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

    def process_image(self, image, b_box_graph, b_box_x, b_box_y, is_x_rotated=False):
        x_min, y_min, x_max, y_max = get_bounding_box(b_box_graph)
        graph = image[y_min:y_max, x_min:x_max]

        x_min_axx, y_min, x_max, y_max = get_bounding_box(b_box_x)
        axis_x = image[y_min:y_max, x_min_axx:x_max]

        x_min_axy, y_min_axy, x_max, y_max = get_bounding_box(b_box_y)
        axis_y = image[y_min_axy:y_max, x_min_axy:x_max]

        graph_contours = self.detect_contours(graph)
        axis_x_contours, text_x = self.detect_text(axis_x, "x", is_x_rotated)
        axis_y_contours, text_y = self.detect_text(axis_y, "y")

        processed_text_x = self.process_detected_text(text_x)
        processed_text_y = self.process_detected_text(text_y)

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
        processed_text_x = sorted(processed_text_x)
        processed_text_y = sorted(processed_text_y)

        if DEBUG:
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
            bound_middle_x += x_min_axx
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
            bound_middle_y += y_min_axy
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

        self.labels_bounding_box = (min_x, min_y, max_x, max_y)
        if DEBUG:
            # Draw bounding box
            cv2.circle(image, (int(min_x), int(max_y)), 5, (255, 255, 0), -1)
            cv2.circle(image, (int(max_x), int(min_y)), 5, (255, 255, 0), -1)
            # Mark on the image the bounding box of the graph
            debug_image(image)
        self.max_x_value = max(processed_text_x)
        self.min_x_value = min(processed_text_x)
        self.max_y_value = max(processed_text_y)
        self.min_y_value = min(processed_text_y)
        if DEBUG:
            print(f"Min x: {self.min_x_value}, Max x: {self.max_x_value}")
            print(f"Min y: {self.min_y_value}, Max y: {self.max_y_value}")

    def detect_text(self, img, var, is_x_rotated=False):
        lista_dados = []
        original_dim = img.shape[:2]
        # Scale img to 300 dpi
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

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

        if var == "x":
            if is_x_rotated:
                # Rotate the image
                rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                custom_config = r"--psm 6 -c tessedit_char_whitelist=0123456789.-ABCDEFHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                altura_x, largura_x, _ = img.shape
                if largura_x < 35:
                    res_x = cv2.resize(rotated, (largura_x + 20, altura_x + 20))
                    extracted_text = pytesseract.image_to_string(
                        res_x, config=custom_config
                    )
                else:
                    extracted_text = pytesseract.image_to_string(
                        rotated, config=custom_config
                    )

                lista_dados.append(extracted_text)
            else:
                custom_config = r"--psm 8 -c tessedit_char_whitelist=0123456789.-ABCDEFHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w < 30:
                        copy_img = img.copy()
                        res_x = cv2.resize(
                            copy_img[y : y + h, x : x + w], (w + 20, h + 20), interpolation=cv2.INTER_CUBIC
                        )
                        extracted_text = pytesseract.image_to_string(res_x, config=custom_config)
                    else:
                        extracted_text = pytesseract.image_to_string(img[y : y + h, x : x + w], config=custom_config)
                    lista_dados.append(extracted_text)

        if var == "y":
            custom_config = r"--psm 6 -c tessedit_char_whitelist=0123456789.-"
            altura_y, largura_y, _ = img.shape
            if largura_y < 35:
                res_y = cv2.resize(img, (largura_y + 20, altura_y + 20))
                extracted_text = pytesseract.image_to_string(
                    res_y, config=custom_config
                )
            else:
                extracted_text = pytesseract.image_to_string(img, config=custom_config)

            lista_dados.append(extracted_text)

        # Transform the countours to the original image dimensions
        dim_diff = np.array(original_dim) / np.array(img.shape[:2])
        contours = [np.array(contour * dim_diff, dtype=np.int32) for contour in contours]
        return contours, lista_dados

    def find_graph_points(self, graph_type: str):
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
            biggest_contour[:, :, 0] += x_min
            biggest_contour[:, :, 1] += y_min
            cv2.drawContours(mask, [biggest_contour], 0, 255, -1)
        elif graph_type == "scatter":
            for contour in contours:
                # Add the padding to the contour, so it is in the original position
                contour[:, :, 0] += x_min
                contour[:, :, 1] += y_min
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
        
        x_array = np.array(sorted_x_coords)
        y_array = np.array(sorted_y_coords)

        # Find unique x values and calculate the mean y for each unique x
        unique_x, mean_y = np.unique(x_array, return_inverse=True)
        mean_y = np.bincount(mean_y, weights=y_array) / np.bincount(mean_y)

        print(f"Unique x: {unique_x.shape}")
        print(f"Mean y: {mean_y.shape}")

        unique_points = []
        for x, y in zip(unique_x, mean_y):
            unique_points.append(Point(x, y))

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
