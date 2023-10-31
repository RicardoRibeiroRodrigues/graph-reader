import os
import cv2
import numpy as np

TRAIN_DIR = os.path.join(os.getcwd() + '/data', 'train/images')
TEST_DIR = os.path.join(os.getcwd() + '/data', 'test/images')
# Get the list of all files in directory tree at given path
list_of_files = [TRAIN_DIR + '/' + file for file in os.listdir(TRAIN_DIR)]

def get_graph_bounding_box(img):
    # Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur the image to reduce noise
    blur = cv2.blur(gray, (3, 3))
    cv2.imshow('Blur', blur)

    # Canny edge detection
    edges = cv2.Canny(blur, 180, 400, apertureSize=3)
    print(edges.shape)

    # Erode
    k = 1
    kernel = np.ones((k, k), np.uint8)
    edges = cv2.erode(edges, kernel, iterations = 1)

    # Dilate
    kernel = np.ones((k, k),np.uint8)
    edges = cv2.dilate(edges, kernel, iterations = 1)

    # Find lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=25)

    # Draw lines
    if lines is not None:
        min_x, max_x, min_y, max_y = np.inf, 0, np.inf, 0
        for line in lines:
            l = line[0]
            print(l)
            x1, y1 = l[0], l[1]
            x2, y2 = l[2], l[3]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            if x2 < min_x:
                min_x = x2
            if x2 > max_x:
                max_x = x2
            if y2 < min_y:
                min_y = y2
            if y2 > max_y:
                max_y = y2
        print(min_x, max_x, min_y, max_y)

        cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2) 
        cv2.imshow('Edges', edges)
        cv2.imshow('Image', img)

for file in list_of_files:
    img = cv2.imread(file)
    get_graph_bounding_box(img)
    if cv2.waitKey(0) == ord('q'):
        break
cv2.destroyAllWindows()
