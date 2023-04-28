import cv2
import mouse
import yaml
import numpy as np
from screeninfo import get_monitors
import HandTrackingModule as Htm

cam_width, cam_height = 1024, 576
screen_width, screen_height = get_monitors()[0].width, get_monitors()[0].height

cap = cv2.VideoCapture(1)
cap.set(3, cam_width)
cap.set(4, cam_height)

crop_offset = 80

detector = Htm.HandDetector(max_hands=1, detection_con=0.2, track_con=0.3)

DIM = (1024, 576)
K = np.array([[797.0118923274562, 0.0, 525.6367078328287],
              [0.0, 800.2609207364642, 280.0906945970121], [0.0, 0.0, 1.0]])
D = np.array([[-0.0031319388008956553], [-0.6096160899954018], [1.981574230203118], [-2.523538885945794]])

# Opening YAML file
with open('dist/RUNTIME DATA/Resources/Corners.yaml', 'r') as openfile:
    # Reading from YAML file
    points = yaml.load(openfile, Loader=yaml.FullLoader)

src = np.float32([(points["Top Fisheye"]["X"], points["Top Fisheye"]["Y"]),
                  (points["Top Right"]["X"], points["Top Right"]["Y"]),
                  (points["Bottom Fisheye"]["X"], points["Bottom Fisheye"]["Y"]),
                  (points["Bottom Right"]["X"], points["Bottom Right"]["Y"])])
mod = np.float32([[crop_offset, crop_offset], [cam_width - crop_offset, crop_offset],
                  [crop_offset, cam_height - crop_offset], [cam_width - crop_offset, cam_height - crop_offset]])
dst = np.float32([[0, 0], [cam_width, 0], [0, cam_height], [cam_width, cam_height]])

matrix = cv2.getPerspectiveTransform(src, dst)
matrix_2 = cv2.getPerspectiveTransform(src, mod)
matrix_fisheye = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, DIM, np.eye(3), balance=0.0)

while True:
    success, img = cap.read()
    if success:
        img = cv2.fisheye.undistortImage(img, K, D, None, Knew=matrix_fisheye)
        img = cv2.resize(img, (1024, 576))
        img = detector.find_hands(img)
        landmarks, bounding_box = detector.find_position(img, draw_lm=False)

        if len(landmarks) != 0:
            landmark = np.array([[landmarks[8][1], landmarks[8][2]]], dtype=np.float32)
            p = (landmarks[8][1], landmarks[8][2])
            px = (matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / (
                (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
            py = (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / (
                (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
            x = np.interp(int(px), (0, cam_width), (0, screen_width))
            y = np.interp(int(py), (0, cam_height), (0, screen_height))
            p_after = (int(x), int(y))
            mouse.move(int(x), int(y))
        for x in range(0, 4):
            cv2.circle(img, (int(src[x][0]), int(src[x][1])), 1, (0, 255, 0), cv2.FILLED)

        img = cv2.warpPerspective(img, matrix_2, DIM)
        cv2.imshow("Original Image", img)
        cv2.waitKey(1)
