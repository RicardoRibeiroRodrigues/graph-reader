import cv2
import numpy as np
import io


def cv2_image_from_bytes(image_data, color_mode=cv2.IMREAD_COLOR):
    # Decode the base64 image data and convert it to a NumPy array
    image_bytes = io.BytesIO(image_data)
    image_array = np.frombuffer(image_bytes.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image_array, color_mode)
    return image


def cv2_image_to_bytes(image):
    # Encode the image as a base64 string
    _, img_encoded = cv2.imencode(".png", image)
    img_base64 = img_encoded.tobytes()
    return img_base64


def debug_image(img):
    cv2.imshow("Image", img)
    while True:
        if cv2.waitKey(0) == ord("q"):
            break
    cv2.destroyAllWindows()


def line_intersec(a1, b1, a2, b2):
    # Solve for x (the intersection point)
    x = (b2 - b1) / (a1 - a2)
    # Substitute x back into one of the equations to find y
    y = a1 * x + b1
    return round(x), round(y)


def get_angle_between_lines(m1, m2):
    div = 1 + m1 * m2
    if div == 0:
        return 0
    angle = np.arctan((m2 - m1) / div)
    angle_deg = np.rad2deg(angle)
    return angle_deg


def perspective_removal(image, source_points, destination_points):
    # Read the image
    original_image = image.copy()
    # Calculate the homography matrix
    homography_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    # Calculate the inverse homography
    inverse_homography_matrix = np.linalg.inv(homography_matrix)
    # Apply the inverse homography to the destination points
    warped_point = cv2.perspectiveTransform(
        np.array([destination_points]), inverse_homography_matrix
    )
    # Apply the homography to obtain the perspective-corrected image
    corrected_image = cv2.warpPerspective(
        original_image,
        homography_matrix,
        (original_image.shape[1], original_image.shape[0]),
    )
    return corrected_image, warped_point[0]


def get_graph_data_norm(img, bbox) -> list:
    x_min, y_min, x_max, y_max = bbox
    graph = img[y_min:y_max, x_min:x_max]
    graph = cv2.cvtColor(graph, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(graph, (11, 11), 0)
    # Performing OTSU threshold
    _, thresh1 = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

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


def ransac(points) -> tuple:
    # RANSAC parameters
    threshold = 0.1
    best_slope = 0
    best_intercept = 0
    best_inliers = []

    for point1 in points:
        for point2 in points:
            if point1 == point2:
                continue
            # Calculate the slope and intercept of the line
            slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
            intercept = point1[1] - slope * point1[0]
            # Calculate the distance of each point to the line
            distances = []
            for point in points:
                distance = (
                    abs(point[1] - slope * point[0] - intercept)
                    / (slope**2 + 1) ** 0.5
                )
                distances.append(distance)
            # Count the number of inliers
            inliers = np.where(np.array(distances) < threshold)[0]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_slope = slope
                best_intercept = intercept
    return best_slope, best_intercept, best_inliers