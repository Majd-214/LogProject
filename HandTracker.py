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

# initialize sensitivity value
sensitivity = 20

# initialize previous touch state
previous_touch_state = False

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
    corners_file = yaml.load(openfile, Loader=yaml.FullLoader)

corners = np.float32([(corners_file['Top Left']['X'], corners_file['Top Left']['Y']),
                      (corners_file['Top Right']['X'], corners_file['Top Right']['Y']),
                      (corners_file['Bottom Left']['X'], corners_file['Bottom Left']['Y']),
                      (corners_file['Bottom Right']['X'], corners_file['Bottom Right']['Y'])])

# Calculate perspective transformation matrices from projection corners
tracking_matrix = cv2.getPerspectiveTransform(corners, dst)

# Initialize hand tracker
detector = Htm.HandDetector(max_hands=1, detection_con=0.3, track_con=0.4)


def get_touch_sensitivity(image, px, py):
    sensitivity_region_size = 20
    y_min = int(max(0, py - sensitivity_region_size / 2))
    y_max = int(min(image.shape[0], py + sensitivity_region_size / 2))
    x_min = int(max(0, px - sensitivity_region_size / 2))
    x_max = int(min(image.shape[1], px + sensitivity_region_size / 2))
    cropped = image[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        return 0.0
    else:
        mean_value = np.mean(cropped)
        if mean_value == 0:
            return 0.0
        else:
            return 1 / mean_value


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

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply thresholding to obtain a binary image
            thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]

            # Find contours in the binary image
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Define a sensitivity threshold
            sensitivity = 127

            # Loop over all contours and check if any have a point with a value greater than the sensitivity threshold
            print('Touch Sensitivity: ' + str(get_touch_sensitivity(gray, int(x), int(y))))

        for x in range(0, 4):
            cv2.circle(img, (int(corners[x][0]), int(corners[x][1])), 1, (0, 255, 0), cv2.FILLED)

        # final_img = cv2.warpPerspective(img, tracking_matrix, (cam_width, cam_height))
        cv2.imshow('Webcam Feed', img)
        cv2.waitKey(1)
