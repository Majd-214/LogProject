import cv2
import mouse
import yaml
import numpy as np
from screeninfo import get_monitors
import HandTrackingModule as Htm

# Burlington Central High School -- TEJ4M1 'The Log Project' --> 'Touch Screen Projector V2' By: Majd Aburas

# Define camera and projection dimensions in pixels
cam_width, cam_height = 1024, 576

# Define the camera indexes chosen
left_camera_index, right_camera_index = 0, 1

# Define the monitor/projector width and height in pixels
screen_width, screen_height = get_monitors()[0].width, get_monitors()[0].height

# Define the crop offset of the projection in camera pixels
crop_offset = 80

# Initialize webcam feeds individually
left_capture = cv2.VideoCapture(left_camera_index)
right_capture = cv2.VideoCapture(right_camera_index)
left_capture.set(3, cam_width)
right_capture.set(3, cam_width)
left_capture.set(4, cam_height)
right_capture.set(4, cam_height)

# Retrieve camera calibration data from YAML file
with open('dist/RUNTIME DATA/Resources/Calibration.yaml', 'r') as f:
    calib_data = yaml.load(f, Loader=yaml.FullLoader)

# Retrieve camera matrix and distortion coefficients for each camera
K_l = np.array(calib_data['camera_0']['camera_matrix'])
D_l = np.array(calib_data['camera_0']['dist_coefficients'])

K_r = np.array(calib_data['camera_1']['camera_matrix'])
D_r = np.array(calib_data['camera_1']['dist_coefficients'])

# Retrieve Stereo calibration data
E = np.array(calib_data['stereo']['E'])
F = np.array(calib_data['stereo']['F'])
P1 = np.array(calib_data['stereo']['P1'])
P2 = np.array(calib_data['stereo']['P2'])
Q = np.array(calib_data['stereo']['Q'])
R = np.array(calib_data['stereo']['R'])
R1 = np.array(calib_data['stereo']['R1'])
R2 = np.array(calib_data['stereo']['R2'])
T = np.array(calib_data['stereo']['T'])

# Calculate the perspective transformation matrix to un-distort webcams' fisheye effect
left_fisheye_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K_l, D_l, (cam_width, cam_height), np.eye(3), balance=0.0)

right_fisheye_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K_r, D_r, (cam_width, cam_height), np.eye(3), balance=0.0)

# Retrieve projection corner data from YAML file
with open('dist/RUNTIME DATA/Resources/Corners.yaml', 'r') as openfile:
    # Reading from YAML file
    corners = yaml.load(openfile, Loader=yaml.FullLoader)

left_corners = np.float32([(corners['Left Camera']['Top Left']['X'], corners['Left Camera']['Top Left']['Y']),
                           (corners['Left Camera']['Top Right']['X'], corners['Left Camera']['Top Right']['Y']),
                           (corners['Left Camera']['Bottom Left']['X'], corners['Left Camera']['Bottom Left']['Y']),
                           (corners['Left Camera']['Bottom Right']['X'], corners['Left Camera']['Bottom Right']['Y'])])

right_corners = np.float32([(corners['Right Camera']['Top Left']['X'], corners['Right Camera']['Top Left']['Y']),
                            (corners['Right Camera']['Top Right']['X'], corners['Right Camera']['Top Right']['Y']),
                            (corners['Right Camera']['Bottom Left']['X'], corners['Right Camera']['Bottom Left']['Y']),
                            (corners['Right Camera']['Bottom Right']['X'], corners['Right Camera']['Bottom Right']['Y'])])

dst = np.float32([[0, 0], [cam_width, 0], [0, cam_height], [cam_width, cam_height]])
mod = np.float32([[crop_offset, crop_offset], [cam_width - crop_offset, crop_offset],
                  [crop_offset, cam_height - crop_offset], [cam_width - crop_offset, cam_height - crop_offset]])

# Calculate perspective transformation matrices from projection corners
left_corner_matrix = cv2.getPerspectiveTransform(left_corners, dst)
right_corner_matrix = cv2.getPerspectiveTransform(right_corners, dst)

# Initialize hand tracker
detector = Htm.HandDetector(max_hands=1, detection_con=0.2, track_con=0.3)

while True:
    # Read webcam feeds
    # left_success, left_img = left_capture.read()
    # right_success, right_img = right_capture.read()
    left_success, left_img = cv2.imread('dist/RUNTIME DATA/Resources/Calibration Images/Stereo/left.jpg')
    right_success, right_img = cv2.imread('dist/RUNTIME DATA/Resources/Calibration Images/Stereo/right.jpg')

    # Apply fisheye distortion removal to left and right separately
    left_img = cv2.fisheye.undistortImage(left_img, K_l, D_l, None, Knew=left_fisheye_matrix)
    right_img = cv2.fisheye.undistortImage(right_img, K_r, D_r, None, Knew=right_fisheye_matrix)
    left_img = cv2.resize(left_img, (cam_width, cam_height))
    right_img = cv2.resize(right_img, (cam_width, cam_height))

    # Initialize the remapping function using the rectification maps
    left_map_X, left_map_Y = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, (cam_width, cam_height), cv2.CV_32FC1)
    right_map_X, right_map_Y = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, (cam_width, cam_height), cv2.CV_32FC1)

    # Apply rectification maps to left and right stereo images
    left_image_rectified = cv2.remap(left_img, left_map_X, left_map_Y, cv2.INTER_LINEAR)
    right_image_rectified = cv2.remap(right_img, right_map_X, right_map_Y, cv2.INTER_LINEAR)

    # Compute the disparity map using StereoBM
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(left_image_rectified, right_image_rectified)

    # Compute depth map
    depth_map = cv2.reprojectImageTo3D(disparity, Q)

    # Calculate homography matrix
    H, _ = cv2.findHomography(dst, left_corners)

    # Transform depth map to projected view
    warped_depth_map = cv2.warpPerspective(depth_map, H, (screen_width, screen_height))

    if left_success:
        left_img = detector.find_hands(left_img)
        landmarks, bounding_box = detector.find_position(left_img, draw_lm=False)

        if len(landmarks) != 0:
            landmark = np.array([[landmarks[8][1], landmarks[8][2]]], dtype=np.float32)
            p = (landmarks[8][1], landmarks[8][2])
            px = (left_corner_matrix[0][0] * p[0] + left_corner_matrix[0][1] * p[1] + left_corner_matrix[0][2]) / (
                (left_corner_matrix[2][0] * p[0] + left_corner_matrix[2][1] * p[1] + left_corner_matrix[2][2]))
            py = (left_corner_matrix[1][0] * p[0] + left_corner_matrix[1][1] * p[1] + left_corner_matrix[1][2]) / (
                (left_corner_matrix[2][0] * p[0] + left_corner_matrix[2][1] * p[1] + left_corner_matrix[2][2]))
            x = np.interp(int(px), (0, cam_width), (0, screen_width))
            y = np.interp(int(py), (0, cam_height), (0, screen_height))
            p_after = (int(x), int(y))
            mouse.move(int(x), int(y))
        for x in range(0, 4):
            cv2.circle(left_img, (int(left_corners[x][0]), int(left_corners[x][1])), 1, (0, 255, 0), cv2.FILLED)

        img = cv2.warpPerspective(left_img, left_corner_matrix, (cam_width, cam_height))
        cv2.imshow('Webcam Feed', depth_map)
        cv2.waitKey(1)
