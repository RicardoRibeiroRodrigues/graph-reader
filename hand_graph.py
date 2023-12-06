import cv2
import numpy as np
import json
from utils import *
import os

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
    

class HandDrawnGraphPipeline:
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

    def threshold_image(self, img, threshold_value=100, blur_amount=3, k=5):
        # Convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Blur the image to reduce noise
        blur = cv2.blur(gray, (blur_amount, blur_amount))
        _, thresh = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY_INV)
        # Erode
        kernel = np.ones((k, k), np.uint8)
        edges = cv2.erode(thresh, kernel, iterations=1)
        # Dilate
        kernel = np.ones((k, k), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        self.threshhold_config = {
            "threshold_value": threshold_value,
            "blur_amount": blur_amount,
            "k": k,
        }
        return cv2_image_to_bytes(edges)

    def find_lines(self, edges, img, line_threshold=60, min_line_percentage=0.25, max_line_gap=3):
        # Find lines
        width = edges.shape[1]
        height = edges.shape[0]
        min_line_length = min(int(min_line_percentage * width), int(min_line_percentage * height))
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, line_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap
        )

        # Find intersections
        if lines is None or len(lines) < 2:
            print("Found no lines")
            return None

        THRESH_LINEAR_COEF = 0.06
        most_orthogonal_lines_angle = 0
        most_orthogonal_lines_intersec = None

        print(f"Number of lines: {len(lines)}")

        for line1 in lines:
            for line2 in lines:
                x1, y1, x2, y2 = line1[0]
                x3, y3, x4, y4 = line2[0]
                if x1 == x2 or x3 == x4:
                    continue
                a1 = (y2 - y1) / (x2 - x1)
                a2 = (y4 - y3) / (x4 - x3)
                b1 = y1 - a1 * x1
                b2 = y3 - a2 * x3
                # don't consider lines with similar slopes
                if abs(a1 - a2) < THRESH_LINEAR_COEF:
                    continue

                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.line(img, (x3, y3), (x4, y4), (255, 0, 0), 2)
                angl = get_angle_between_lines(a1, a2)
                if abs(angl - 90) < abs(most_orthogonal_lines_angle - 90):
                    x, y = line_intersec(a1, b1, a2, b2)
                    if x < 0 or x > width or y < 0 or y > height:
                        continue
                    most_orthogonal_lines_angle = angl
                    most_orthogonal_lines_intersec = (x, y)
                    cv2.circle(img, (x, y), 15, (0, 255, 255), -1)

        if most_orthogonal_lines_intersec is None:
            print("Found no line intersections")
            return None
        
        self.lines_intersec = most_orthogonal_lines_intersec
        return cv2_image_to_bytes(img)


    def find_handdrawn_bbox(self, img, thresholded, pad_size_x=30, pad_size_y=30) -> list:
        original_img = img.copy()
        most_orthogonal_lines_intersec = self.lines_intersec

        # Lowest y value in the mask -> y_min
        # Highest x value in the mask -> x_max
        # Intersections -> Graph origin -> (x_min, y_max)
        valid_graph_points = np.where(thresholded == 255)
        y_min = np.min(valid_graph_points[0])
        x_min = most_orthogonal_lines_intersec[0]
        y_max = most_orthogonal_lines_intersec[1]
        x_max = np.max(valid_graph_points[1])
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # X min of the axis -> x_min_axis
        x_min_axis = np.min(valid_graph_points[1])
        # Find the point of the minimum y value in the image
        top_point = np.argmin(valid_graph_points[0])
        top_point = (valid_graph_points[1][top_point], valid_graph_points[0][top_point])
        # Bounding box of the graph -> (x_min, y_min, x_max, y_max)
        bbox_1 = (x_min, y_min, x_max, y_max)
        # Bounding box of the graph + axis -> (x_min_axis, y_min, x_max, y_max)
        bbox_2 = (x_min_axis, y_min, x_max, y_max)
        source_points_1 = np.float32(
            [
                top_point,
                [bbox_1[2], bbox_1[1]],
                [bbox_1[0], bbox_1[3]],
                [bbox_1[2], bbox_1[3]],
            ]
        )
        dest_points = np.float32(
            [
                [bbox_2[0], bbox_2[1]],
                [bbox_2[2], bbox_2[1]],
                [bbox_2[0], bbox_2[3]],
                [bbox_2[2], bbox_2[3]],
            ]
        )

        # Concatenate the source points of both bounding boxes
        corrected_image, new_bbox = perspective_removal(
            original_img, source_points_1, dest_points
        )
        
        # draw destination points as red circles
        min_x = min_y = np.Infinity
        max_x = max_y = 0
        # for point in new_bbox:
        for point in dest_points:
            cv2.circle(corrected_image, (int(point[0]), int(point[1])), 15, (0, 0, 255), -1)
            min_x = int(min(min_x, point[0]))
            min_y = int(min(min_y, point[1]))
            max_x = int(max(max_x, point[0]))
            max_y = int(max(max_y, point[1]))

        # Pad the bounding box to exclude the axis
        self.found_bbox_padded = (
            min_x + pad_size_x,
            min_y + pad_size_y,
            max_x - pad_size_x,
            max_y - pad_size_y,
        )
        cv2.rectangle(
            corrected_image,
            (self.found_bbox_padded[0], self.found_bbox_padded[1]),
            (self.found_bbox_padded[2], self.found_bbox_padded[3]),
            (0, 255, 0),
            2,
        )
        self.found_bbox = (min_x, min_y, max_x, max_y)
        self.corrected_image = corrected_image
        return cv2_image_to_bytes(corrected_image)

    def find_graph_points(self, img, graph_type: str):
        print(f"Graph type: {graph_type}")
        if graph_type not in ("line", "scatter"):
            return None
        
        # Find the The thresholded image again, with corrected image.
        res = self.threshold_image(img, **self.threshhold_config)
        th = cv2_image_from_bytes(res, cv2.IMREAD_GRAYSCALE)
        thresholded_crop = th[
            self.found_bbox_padded[1] : self.found_bbox_padded[3],
            self.found_bbox_padded[0] : self.found_bbox_padded[2],
        ]

        contours, _ = cv2.findContours(
            thresholded_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            print("Found no contours")
            return None

        mask = np.zeros(th.shape, np.uint8)
        # If is a line plot, consider only the biggest line (using contour area)
        if graph_type == "line":
            biggest_contour = max(contours, key=cv2.contourArea)
            # Add the padding to the contour, so it is in the original position
            biggest_contour[:, :, 0] += self.found_bbox_padded[0]
            biggest_contour[:, :, 1] += self.found_bbox_padded[1]
            cv2.drawContours(img, [biggest_contour], 0, (0, 0, 255), 2)
            cv2.drawContours(mask, [biggest_contour], 0, 255, -1)
            # Get the bounding box of the contour
            # x_rect, y_rect, w, h = cv2.boundingRect(biggest_contour)
            # cv2.rectangle(img, (x_rect, y_rect), (x_rect + w, y_rect + h), (255, 255, 0), 2)
            # # Crop the image to the bounding box
            # img_th = th[y_rect : y_rect + h, x_rect : x_rect + w]
        elif graph_type == "scatter":
            # If is a scatter plot, consider all the contours
            avg_area = sum([cv2.contourArea(contour) for contour in contours]) / len(contours)
            print("Average area: ", avg_area)
            for contour in contours:
                if cv2.contourArea(contour) > avg_area * 1.3:
                    continue
                # Add the padding to the contour, so it is in the original position
                contour[:, :, 0] += self.found_bbox_padded[0]
                contour[:, :, 1] += self.found_bbox_padded[1]
                cv2.drawContours(img, [contour], 0, (0, 0, 255), 2)
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
            # print(sorted_y_coords[i])
            unique_points.append(Point(sorted_x_coords[i], sorted_y_coords[i]))

        # # Remove duplicate x coordinates
        # unique_points = set()
        # for x, y in zip(x_coords, y_coords):
        #     point = Point(x, y)
        #     unique_points.add(point)

        # for x in range(0, img_th.shape[1], 2):
        #     avg_y_position = 0
        #     count_points = 0
        #     for y in range(img_th.shape[0]):
        #         if img_th[y, x] == 255:
        #             avg_y_position += y
        #             count_points += 1
            
        #     if count_points != 0:
        #         avg_y_position /= count_points
        #         unique_x_coords.append(x + x_rect)
        #         unique_y_coords.append(avg_y_position + y_rect)
        str_points = []
        for point in unique_points:
            x, y = self.normalize_point(img.shape, point.x, point.y)
            str_points.append(f"{x},{y}\n")
        
        csv_str = "".join(str_points)
        # DEBUG
        if DEBUG:
            with open("out.csv", 'w') as f:
                f.write(csv_str)
        
        return csv_str.encode("utf-8")

        
    
