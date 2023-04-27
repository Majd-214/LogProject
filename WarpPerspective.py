import cv2
import yaml
import numpy as np
import CalibrationUtilities as Utils
from screeninfo import get_monitors

cam_width, cam_height = 1024, 576
screen_width, screen_height = get_monitors()[0].width, get_monitors()[0].height

cap = cv2.VideoCapture(1)
cap.set(3, cam_width)
cap.set(4, cam_height)

DIM = (1024, 576)
K = np.array([[797.0118923274562, 0.0, 525.6367078328287],
              [0.0, 800.2609207364642, 280.0906945970121], [0.0, 0.0, 1.0]])
D = np.array([[-0.0031319388008956553], [-0.6096160899954018], [1.981574230203118], [-2.523538885945794]])

dst = np.float32([[0, 0], [cam_width, 0], [0, cam_height], [cam_width, cam_height]])

matrix_fisheye = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, DIM, np.eye(3), balance=0.0)

Utils.initialize_trackbars()

while True:
    success, img = cap.read()
    img = cv2.fisheye.undistortImage(img, K, D, None, Knew=matrix_fisheye)
    img = cv2.resize(img, (cam_width, cam_height))

    # imgBlank = np.zeros((cam_width, cam_height, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGGING IF
    # REQUIRED
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    threshold = Utils.val_trackbars()  # GET TRACK BAR VALUES FOR THRESHOLDS
    imgThreshold = cv2.Canny(imgBlur, threshold[0], threshold[1])  # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION

    # FIND ALL CONTOURS
    imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS

    # FIND THE BIGGEST CONTOUR
    biggest, maxArea = Utils.biggest_contour(contours)  # FIND THE BIGGEST CONTOUR

    if biggest.size != 0:
        biggest = Utils.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
        imgBigContour = Utils.draw_rectangle(imgBigContour, biggest, 2)
        src = np.float32(biggest)
        matrix = cv2.getPerspectiveTransform(src, dst)
        # imgBigContour = cv2.warpPerspective(imgBigContour, matrix, DIM)

    cv2.imshow("Original Image", imgBigContour)
    # SAVE IMAGE WHEN 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        data = {
            'Top Left': {'X': int(biggest[0][0][0]), 'Y': int(biggest[0][0][1])},
            'Top Right': {'X': int(biggest[1][0][0]), 'Y': int(biggest[1][0][1])},
            'Bottom Left': {'X': int(biggest[2][0][0]), 'Y': int(biggest[2][0][1])},
            'Bottom Right': {'X': int(biggest[3][0][0]), 'Y': int(biggest[3][0][1])},
        }

        # Writing to Corners.yaml
        with open("dist/RUNTIME DATA/Resources/Corners.yaml", "w") as outfile:
            yaml.dump(data, outfile, sort_keys=False)

        # Confirm saved parameters
        print("SAVED!")

        # Exit program
        break
