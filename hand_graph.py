import cv2
import numpy as np
from utils import *
import os

DEBUG = os.environ.get("DEBUG", False)


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

    
    def normalize_point(self, x, y):
        min_y = self.found_bbox[1]
        min_x = self.found_bbox[0]
        max_y = self.found_bbox[3]
        max_x = self.found_bbox[2]
        
        # Normalize the coordinates
        normalized_x = (x - min_x) / (max_x - min_x)
        normalized_y = (y - min_y) / (max_y - min_y)
        normalized_y = 1 - normalized_y
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


    def find_handdrawn_bbox(self, img, thresholded, pad_size_x_percent=0.15, pad_size_y_percent=0.15) -> list:
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
        corrected_image, _ = perspective_removal(
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
        pad_size_x = int((max_x - min_x) * pad_size_x_percent)
        pad_size_y = int((max_y - min_y) * pad_size_y_percent)
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
        elif graph_type == "scatter":
            # If is a scatter plot, consider all the contours
            avg_area = sum([cv2.contourArea(contour) for contour in contours]) / len(contours)
            print("Average area: ", avg_area)
            for contour in contours:
                # Ignore contours that are too big (probably some part of the axis)
                if cv2.contourArea(contour) > avg_area * 1.5:
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

        x_array = np.array(sorted_x_coords)
        y_array = np.array(sorted_y_coords)

        # Find unique x values and calculate the mean y for each unique x
        unique_x, mean_y = np.unique(x_array, return_inverse=True)
        mean_y = np.bincount(mean_y, weights=y_array) / np.bincount(mean_y)

        unique_points = [Point(x, y) for x, y in zip(unique_x, mean_y)]

        str_points = []
        for point in unique_points:
            x, y = self.normalize_point(point.x, point.y)
            str_points.append(f"{x},{y}\n")
        
        csv_str = "".join(str_points)
        # DEBUG
        if DEBUG:
            with open("out.csv", 'w') as f:
                f.write(csv_str)
        
        return csv_str.encode("utf-8")

        
    
