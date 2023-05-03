import mouse
import yaml
import numpy as np
import cv2
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

# Retrieve camera fisheye lens distortion coefficients
K_l = np.array(calib_data['camera_0']['K'])
D_l = np.array(calib_data['camera_0']['D'])
K_r = np.array(calib_data['camera_1']['K'])
D_r = np.array(calib_data['camera_1']['D'])

# Retrieve camera intrinsic and extrinsic calibration parameters
left_camera_matrix = np.array(calib_data['camera_0']['camera_matrix'])
left_distortion_coefficients = np.array(calib_data['camera_0']['dist_coefficients'])
right_camera_matrix = np.array(calib_data['camera_1']['camera_matrix'])
right_distortion_coefficients = np.array(calib_data['camera_1']['dist_coefficients'])

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
projection_matrix = cv2.getPerspectiveTransform(dst, contour)
tracking_matrix = cv2.getPerspectiveTransform(contour, dst)

# Initialize hand tracker
detector = Htm.HandDetector(max_hands=1, detection_con=0.2, track_con=0.3)

while True:
    # Read webcam feeds
    left_success, left_img = left_capture.read()
    right_success, right_img = right_capture.read()

    # Apply fisheye distortion removal to left and right separately
    left_img = cv2.fisheye.undistortImage(left_img, K_l, D_l, None, Knew=left_fisheye_matrix)
    right_img = cv2.fisheye.undistortImage(right_img, K_r, D_r, None, Knew=right_fisheye_matrix)
    left_img = cv2.resize(left_img, (cam_width, cam_height))
    right_img = cv2.resize(right_img, (cam_width, cam_height))

    # Compute the disparity map
    window_size = 5
    min_disp = 0
    num_disp = 64
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=window_size,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32,
                                   disp12MaxDiff=1,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2)
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

    # Generate the 3D point cloud
    focal_length = left_camera_matrix[0][0]
    Q = np.float32([[1, 0, 0, -left_camera_matrix[0][2]],
                    [0, -1, 0, left_camera_matrix[1][2]],
                    [0, 0, 0, -focal_length],
                    [0, 0, 1, 0]])
    point_cloud = cv2.reprojectImageTo3D(disparity, Q)
    depth_map = point_cloud[:, :, 2]

    # Create a 3D array with x, y, and z coordinates of each pixel
    h, w = depth_map.shape[:2]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    ones = np.ones_like(X)
    XYZ = np.stack((X, Y, ones), axis=-1)
    XYZ = np.expand_dims(depth_map, axis=-1) * XYZ

    # Apply the perspective transformation to the x, y, and z coordinates separately
    XYZ_transformed = cv2.perspectiveTransform(XYZ.reshape(-1, 1, 3), projection_matrix).reshape(h, w, 3)

    # Extract the z-coordinate (depth) values from the transformed coordinates
    depth_map_transformed = XYZ_transformed[:, :, 2]

    if left_success:
        left_img = detector.find_hands(left_img)
        landmarks, bounding_box = detector.find_position(left_img, draw_lm=False)

        if len(landmarks) != 0:
            landmark = np.array([[landmarks[8][1], landmarks[8][2]]], dtype=np.float32)
            p = (landmarks[8][1], landmarks[8][2])
            px = (projection_matrix[0][0] * p[0] + projection_matrix[0][1] * p[1] + projection_matrix[0][2]) / (
                (projection_matrix[2][0] * p[0] + projection_matrix[2][1] * p[1] + projection_matrix[2][2]))
            py = (projection_matrix[1][0] * p[0] + projection_matrix[1][1] * p[1] + projection_matrix[1][2]) / (
                (projection_matrix[2][0] * p[0] + projection_matrix[2][1] * p[1] + projection_matrix[2][2]))
            x = np.interp(int(px), (0, cam_width), (0, screen_width))
            y = np.interp(int(py), (0, cam_height), (0, screen_height))
            p_after = (int(x), int(y))
            mouse.move(int(x), int(y))
            print(depth_map_transformed[p[1]][p[0]])
        for x in range(0, 4):
            cv2.circle(left_img, (int(contour[x][0]), int(contour[x][1])), 1, (0, 255, 0), cv2.FILLED)

        img = cv2.warpPerspective(left_img, projection_matrix, (cam_width, cam_height))
        cv2.imshow('Webcam Feed', depth_map)
        cv2.waitKey(1)
