import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def read_image(path):
    original_img = cv.imread(path)
    original_img = cv.resize(original_img, (int(original_img.shape[1] * 0.3), int(original_img.shape[0] * 0.3)), interpolation=cv.INTER_AREA)
    original_img = cv.rotate(original_img, cv.ROTATE_90_COUNTERCLOCKWISE)

    img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5, 5), 0)
    
    return original_img, img

def read_frame(frame):
    original_img = cv.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)), interpolation=cv.INTER_AREA)
    original_img = frame
    img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5, 5), 0)
    
    return original_img, img

def scoresheet_border(img):
    largest_contour = None

    ret, thresh = cv.threshold(img, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(img, contours, -1, (0, 255, 0), thickness=3)
    # cv.imshow("e", img)
    # cv.waitKey(0)

    for contour in contours:
        if largest_contour is None or (cv.arcLength(contour,True) > cv.arcLength(largest_contour,True)):
            largest_contour = contour
    
    if largest_contour is not None:
        epsilon = 0.1*cv.arcLength(largest_contour,True)
        approx = cv.approxPolyDP(largest_contour,epsilon,True)

        # cv.drawContours(img, [approx], 0, (0, 255, 0), 3)
        # cv.imshow("e", img)

        return approx
    else:
        return None

def order_points(pts):
    print(pts)
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

def find_targets(img):
    img_blur = cv.medianBlur(img, 21)

    rows = img.shape[0]
    circles = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=30, minRadius=1, maxRadius=100)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])

            # circle center
            #cv.circle(img, center, 1, (0, 100, 100), 3)

            # circle outline
            radius = i[2]
            #cv.circle(img, center, radius, (255, 0, 0), 3)

            # cropped_img = img[int(i[1] - radius):int(i[1] + radius), int(i[0] - radius):int(i[0] + radius)]
            # cv.imshow("e", cropped_img)
    else:
        print("No circles found")

    return img

def find_shots(img):
    #cv.imshow("og", img)
    #img = img[int(img.shape[0] * 1/8):int(img.shape[0] - img.shape[0] * 1/16), :]
    img = img[:, int(img.shape[1] * 1/12):int(img.shape[1] - img.shape[1] * 1/12)]
    copy_img = cv.GaussianBlur(img, (5,5), 3)
    copy_img = cv.cvtColor(copy_img, cv.COLOR_BGRA2GRAY)
    copy_img = cv.Canny(copy_img, 50, 150)
    #cv.imshow("edited", copy_img)

    ret, thresh = cv.threshold(copy_img, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x,y),radius = cv.minEnclosingCircle(contour)
        center = (int(x),int(y))
        radius = int(radius)
        
        if 4 <= radius <= 5 and 2 * np.pi * radius - cv.arcLength(contour, True) < 10:
            cv.circle(img, center, radius, (0, 255, 0), 1)
        #elif radius >= 15:
            #cv.drawContours(copy_img, [contour], 0, (0, 0, 0), 1)

    return copy_img, img

source = cv.VideoCapture(1)
source.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
source.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

win_name = "Camera Preview"
cv.namedWindow(win_name, cv.WINDOW_NORMAL)

while not(cv.waitKey(1) & 0xFF == ord('q')):
    has_frame, frame = source.read()
    if not has_frame:
        break
    original_img, img = read_frame(frame)

    border = scoresheet_border(img)
    print(border)
    if border is not None:
        if len(border) == 4:
            warped = warped_scoresheet(original_img, border)
            scoresheet = find_targets(cv.cvtColor(warped, cv.COLOR_BGR2GRAY))
            scoresheet = cv.cvtColor(scoresheet, cv.COLOR_GRAY2BGR)
            copy_img, img = find_shots(scoresheet)
            cv.imshow(win_name + "-2", copy_img)
            cv.imshow(win_name, img)
        else:
            cv.drawContours(frame, [border], 0, (0, 255, 0), 3)
            cv.imshow(win_name, frame)

source.release()
cv.destroyAllWindows()

# original_img, img = read_image("Photos/scoresheet_12.jpg")
# border = scoresheet_border(img)

# warped = warped_scoresheet(original_img, border)
# scoresheet = find_targets(cv.cvtColor(warped, cv.COLOR_BGR2GRAY))
# scoresheet = cv.cvtColor(scoresheet, cv.COLOR_GRAY2BGR)
# copy_img, img = find_shots(scoresheet)
# cv.imshow("Scoresheet", img)

# cv.waitKey(0)