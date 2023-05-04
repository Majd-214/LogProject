import mouse
import yaml
import numpy as np
import cv2
from screeninfo import get_monitors
import HandTrackingModule as Htm

# Burlington Central High School -- TEJ4M1 'The Log Project' --> 'Touch Screen Projector V2' By: Majd Aburas

# Define camera and projection dimensions in pixels
cam_width, cam_height = 1920, 1080

# Define desired dimensions
width, height = 1024, 576

# Define maximum frame rate
max_frame_rate = 20

# Define the camera indexes chosen
camera_index = 0

# Define the monitor/projector width and height in pixels
screen_width, screen_height = get_monitors()[0].width, get_monitors()[0].height

# Define the crop offset of the projection in camera pixels
crop_offset = 80

# Initialize webcam feeds individually
capture = cv2.VideoCapture(camera_index)
capture.set(3, cam_width)
capture.set(4, cam_height)

# Retrieve camera calibration data from YAML file
with open('dist/RUNTIME DATA/Resources/Calibration.yaml', 'r') as f:
    calib_data = yaml.load(f, Loader=yaml.FullLoader)

# Retrieve camera fisheye lens distortion coefficients
K = np.array(calib_data['camera_0']['K'])
D = np.array(calib_data['camera_0']['D'])

# Calculate the perspective transformation matrix to un-distort webcams' fisheye effect
fisheye_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K, D, (cam_width, cam_height), np.eye(3), balance=0.0)

dst = np.float32([[0, 0], [cam_width, 0], [0, cam_height], [cam_width, cam_height]])
mod = np.float32([[crop_offset, crop_offset], [cam_width - crop_offset, crop_offset],
                  [crop_offset, cam_height - crop_offset], [cam_width - crop_offset, cam_height - crop_offset]])

# Retrieve projection corner data from YAML file
with open('dist/RUNTIME DATA/Resources/Corners.yaml', 'r') as openfile:
    # Reading from YAML file
    corners = yaml.load(openfile, Loader=yaml.FullLoader)

contour = np.float32([(corners['Top Left']['X'], corners['Top Left']['Y']),
                      (corners['Top Right']['X'], corners['Top Right']['Y']),
                      (corners['Bottom Left']['X'], corners['Bottom Left']['Y']),
                      (corners['Bottom Right']['X'], corners['Bottom Right']['Y'])])

# Calculate perspective transformation matrices from projection corners
tracking_matrix = cv2.getPerspectiveTransform(contour, dst)

# Initialize hand tracker
detector = Htm.HandDetector(max_hands=1, detection_con=0.3, track_con=0.4)

while True:
    # Read webcam feeds
    success, img = capture.read()

    # Apply fisheye distortion removal to left and right separately
    img = cv2.fisheye.undistortImage(img, K, D, None, Knew=fisheye_matrix)
    img = cv2.resize(img, (width, height))

    if success:
        img = detector.find_hands(img)
        landmarks, bounding_box = detector.find_position(img, draw_lm=False)

        if len(landmarks) != 0:
            landmark = np.array([[[landmarks[8][1], landmarks[8][2]]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(landmark, tracking_matrix)
            x = np.interp(transformed[0][0][0], (0, cam_width), (0, screen_width))
            y = np.interp(transformed[0][0][1], (0, cam_height), (0, screen_height))
            z = landmarks[8][3]
            mouse.move(int(x), int(y))
            if detector.fingers_up()[0] == 0:
                mouse.click()
        for x in range(0, 4):
            cv2.circle(img, (int(contour[x][0]), int(contour[x][1])), 1, (0, 255, 0), cv2.FILLED)

        # final_img = cv2.warpPerspective(img, tracking_matrix, (cam_width, cam_height))
        cv2.imshow('Webcam Feed', img)
        cv2.waitKey(1)
