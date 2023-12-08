import cv2
import numpy as np
from utils import *
import os

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

    def find_graph_points(self, img, graph_type: str):
        print(f"Graph type: {graph_type}")
        if graph_type not in ("line", "scatter"):
            return None

        # Find the The thresholded image again, with corrected image.
        # Format x_min, y_min, x_max, y_max
        x_min = self.graph_bounding_box[0]
        y_min = self.graph_bounding_box[1]
        x_max = self.graph_bounding_box[2]
        y_max = self.graph_bounding_box[3]
        thresholded_crop = self.thresholded_image[
            y_min:y_max, x_min:x_max
        ]
            

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
            if (mask.shape > (1000, 1000)):
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
        x_min = self.graph_bounding_box[0]
        y_min = self.graph_bounding_box[1]
        x_max = self.graph_bounding_box[2]
        y_max = self.graph_bounding_box[3]
        # Calculate the multipliers for the normalized coordinates to the graph labels interval
        x_multiplier = (self.max_x_value - self.min_x_value) 
        y_multiplier = (self.max_y_value - self.min_y_value)

        for point in unique_points:
            x, y = self.normalize_point(point.x, point.y)
            # Translate the normalized coordinates to the graph labels interval
            x = self.min_x_value + x * x_multiplier
            y = self.min_y_value + y * y_multiplier
            str_points.append(f"{x},{y}\n")

        csv_str = "".join(str_points)
        # DEBUG
        if DEBUG:
            with open("out.csv", 'w') as f:
                f.write(csv_str)

        return csv_str.encode("utf-8")
