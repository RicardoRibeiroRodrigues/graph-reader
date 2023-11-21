import cv2
import numpy as np
from math import cos, sin

def debug_image(img):
    cv2.imshow('Image', img)
    while True:
        if cv2.waitKey(0) == ord('q'):
            break
    cv2.destroyAllWindows()


def hough_inter(theta1, rho1, theta2, rho2):
    A = np.array([[cos(theta1), sin(theta1)], 
                  [cos(theta2), sin(theta2)]])
    b = np.array([rho1, rho2])
    return np.linalg.lstsq(A, b)[0] # use lstsq to solve Ax = b, not inv() which is unstable

def line_intersec(a1, b1, a2, b2):
    # Solve for x (the intersection point)
    x = (b2 - b1) / (a1 - a2)

    # Substitute x back into one of the equations to find y
    y = a1 * x + b1

    return round(x), round(y)

def find_handdrawn_bbox(img) -> list:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur the image to reduce noise
    blur = cv2.blur(img_gray, (3, 3))
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Canny edge detection
    # edges = cv2.Canny(blur, 280, 400, apertureSize=3)

    # Erode
    k = 1
    kernel = np.ones((k, k), np.uint8)
    edges = cv2.erode(thresh, kernel, iterations = 1)

    # Dilate
    kernel = np.ones((k, k),np.uint8)
    edges = cv2.dilate(thresh, kernel, iterations = 1)

    # Find lines
    width = img.shape[1]
    height = img.shape[0]
    min_line_length = min(int(0.6 * width), int(0.6 * height))
    # lines = cv2.HoughLines(edges, 1, np.pi/180, 240)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 500, minLineLength=min_line_length, maxLineGap=3)

    # Find intersections
    intersections = []
    if lines is None:
        return intersections
    
    for line1 in lines:
        for line2 in lines:
            x1, y1, x2, y2 = line1[0]
            x3, y3, x4, y4 = line2[0]
            a1 = (y2 - y1) / (x2 - x1)
            a2 = (y4 - y3) / (x4 - x3)
            b1 = y1 - a1 * x1
            b2 = y3 - a2 * x3
            if abs(a1 - a2) < 0.001:
                continue
            
            cv2.line(img,(x1,y1),(x2,y2),(255,0, 0),2)
            cv2.line(img,(x3,y3),(x4,y4),(255,0, 0),2)
            x, y = line_intersec(a1, b1, a2, b2)
            if x < 0 or x > width or y < 0 or y > height:
                continue
            intersections.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    
    # Lowest y value in the mask -> y_min
    # Highest x value in the mask -> x_max
    # Intersections -> Graph origin -> (x_min, y_max)
    valid_graph_points = np.where(thresh == 255)
    y_min = np.min(valid_graph_points[0])
    x_min = intersections[0][0]
    y_max = intersections[0][1]
    x_max = np.max(valid_graph_points[1])
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    # X min of the axis -> x_min_axis
    x_min_axis = np.min(valid_graph_points[1])
    # Bounding box of the graph -> (x_min, y_min, x_max, y_max)
    bbox_1 = (x_min, y_min, x_max, y_max)
    # Bounding box of the graph + axis -> (x_min_axis, y_min, x_max, y_max)
    bbox_2 = (x_min_axis, y_min, x_max, y_max)
    source_points_1 = np.float32([[bbox_1[0], bbox_1[1]], [bbox_1[2], bbox_1[1]], [bbox_1[0], bbox_1[3]], [bbox_1[2], bbox_1[3]]])
    source_points_2 = np.float32([[bbox_2[0], bbox_2[1]], [bbox_2[2], bbox_2[1]], [bbox_2[0], bbox_2[3]], [bbox_2[2], bbox_2[3]]])

    # Concatenate the source points of both bounding boxes
    corrected_image = perspective_removal(img, source_points_1, source_points_2)
    debug_image(corrected_image)
    return bbox_1

def perspective_removal(image, source_points, destination_points):
    # Read the image
    original_image = image.copy()

    # Convert points to NumPy arrays
    source_points = np.float32(source_points)
    destination_points = np.float32(destination_points)

    # Calculate the homography matrix
    homography_matrix = cv2.getPerspectiveTransform(source_points, destination_points)

    # Apply the homography to obtain the perspective-corrected image
    corrected_image = cv2.warpPerspective(original_image, homography_matrix, (original_image.shape[1], original_image.shape[0]))

    return corrected_image


def get_graph_data_norm(img, bbox) -> list:
    x_min, y_min, x_max, y_max = bbox
    graph = img[y_min:y_max, x_min:x_max]
    graph = cv2.cvtColor(graph, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(graph, (11, 11), 0)
    # Performing OTSU threshold
    _, thresh1 = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

    graph_points = []
    
    # Loop points of theshold image
    white_points = np.where(dilation == 255)
    x_points, y_points = white_points[1], white_points[0]
    # Map x_points and y_points and normalize them
    x_points = (x_points - x_min) / (x_max - x_min)
    y_points = (y_points - y_min) / (y_max - y_min)
    for x, y in zip(x_points, y_points):
        graph_points.append((x, y))
    
    return graph_points