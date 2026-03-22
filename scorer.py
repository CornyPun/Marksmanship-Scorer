import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def read_image(path):
    original_img = cv.imread(path)
    original_img = cv.resize(original_img, (int(original_img.shape[1] * 0.5), int(original_img.shape[0] * 0.5)), interpolation=cv.INTER_AREA)
    
    img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5, 5), 0)
    
    return original_img, img

def scoresheet_border(img):
    largest_contour = None

    ret, thresh = cv.threshold(img, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if largest_contour is None or cv.arcLength(contour,True) > cv.arcLength(largest_contour,True):
            largest_contour = contour
            
    epsilon = 0.1*cv.arcLength(largest_contour,True)
    approx = cv.approxPolyDP(largest_contour,epsilon,True)

    return approx

def order_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect

def warped_scoresheet(img, border):
    rect = order_points(border)

    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(img, M, (maxWidth, maxHeight))

    return warped

def find_circles(img):
    img_blur = cv.medianBlur(img, 21)

    rows = img.shape[0]
    circles = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=30, minRadius=1, maxRadius=100)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])

            # circle center
            cv.circle(img, center, 1, (0, 100, 100), 3)

            # circle outline
            radius = i[2]
            cv.circle(img, center, radius, (255, 0, 0), 3)
    else:
        print("No circles found")

    return img

original_img, img = read_image("Photos/scoresheet_2.jpg")
original_img = cv.rotate(original_img, cv.ROTATE_90_COUNTERCLOCKWISE)
img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)

border = scoresheet_border(img)
warped = warped_scoresheet(original_img, border)

cv.imshow("Scoresheet", find_circles(cv.cvtColor(warped, cv.COLOR_BGR2GRAY)))

cv.waitKey(0)