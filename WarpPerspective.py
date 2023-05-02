import yaml
import numpy as np
import CalibrationUtilities as Utils
import cv2
from screeninfo import get_monitors

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
K_l = np.array(calib_data['camera_0']['K'])
D_l = np.array(calib_data['camera_0']['D'])
K_r = np.array(calib_data['camera_1']['K'])
D_r = np.array(calib_data['camera_1']['D'])

# Calculate the perspective transformation matrix to un-distort webcams' fisheye effect
left_fisheye_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K_l, D_l, (cam_width, cam_height), np.eye(3), balance=0.0)

right_fisheye_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K_r, D_r, (cam_width, cam_height), np.eye(3), balance=0.0)

dst = np.float32([[0, 0], [cam_width, 0], [0, cam_height], [cam_width, cam_height]])
mod = np.float32([[crop_offset, crop_offset], [cam_width - crop_offset, crop_offset],
                  [crop_offset, cam_height - crop_offset], [cam_width - crop_offset, cam_height - crop_offset]])

# Initialize trackbars for contour filtering
Utils.initialize_trackbars()

while True:
    # Read webcam feeds
    left_success, left_img = left_capture.read()
    right_success, right_img = right_capture.read()

    # Apply fisheye distortion removal to left and right separately
    left_img = cv2.fisheye.undistortImage(left_img, K_l, D_l, None, Knew=left_fisheye_matrix)
    right_img = cv2.fisheye.undistortImage(right_img, K_r, D_r, None, Knew=right_fisheye_matrix)
    left_img = cv2.resize(left_img, (cam_width, cam_height))
    right_img = cv2.resize(right_img, (cam_width, cam_height))

    # Retrieve trackbar values for thresholding
    threshold = Utils.val_trackbars()
    kernel = np.ones((5, 5))

    # Apply image filters to allow for contour filtering
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)  # Convert image to Grayscale
    left_blur = cv2.GaussianBlur(left_gray, (5, 5), 1)  # Add Gaussian blur
    left_threshold = cv2.Canny(left_blur, threshold[0], threshold[1])  # Apply Canny blur
    left_dilation = cv2.dilate(left_threshold, kernel, iterations=2)  # Apply Dilation
    left_threshold = cv2.erode(left_dilation, kernel, iterations=1)  # Apply Erosion

    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)  # Convert image to Grayscale
    right_blur = cv2.GaussianBlur(right_gray, (5, 5), 1)  # Add Gaussian blur
    right_threshold = cv2.Canny(right_blur, threshold[0], threshold[1])  # Apply Canny blur
    right_dilation = cv2.dilate(right_threshold, kernel, iterations=2)  # Apply Dilation
    right_threshold = cv2.erode(right_dilation, kernel, iterations=1)  # Apply Erosion

    # Find all contours in Image
    left_largest_contour = left_img.copy()  # Copy image for display purposes
    left_contours, left_hierarchy = cv2.findContours(left_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    right_largest_contour = right_img.copy()  # Copy image for display purposes
    right_contours, right_hierarchy = cv2.findContours(right_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Isolate the largest contour by area
    left_largest, left_max_area = Utils.biggest_contour(left_contours)
    right_largest, right_max_area = Utils.biggest_contour(right_contours)

    # Draw the largest contour found
    if left_largest.size != 0:
        left_largest = Utils.reorder(left_largest)
        cv2.drawContours(left_largest_contour, left_largest, -1, (0, 255, 0), 20)  # Draw the largest contour
        left_largest_contour = Utils.draw_rectangle(left_largest_contour, left_largest, 2)
        left_src = np.float32(left_largest)
        left_matrix = cv2.getPerspectiveTransform(left_src, dst)
        # left_largest_contour = cv2.warpPerspective(left_largest_contour, left_matrix, (cam_width, cam_height))

    if right_largest.size != 0:
        right_largest = Utils.reorder(right_largest)
        cv2.drawContours(right_largest_contour, right_largest, -1, (0, 255, 0), 20)  # Draw the largest contour
        right_largest_contour = Utils.draw_rectangle(right_largest_contour, right_largest, 2)
        right_src = np.float32(right_largest)
        right_matrix = cv2.getPerspectiveTransform(right_src, dst)
        # right_largest_contour = cv2.warpPerspective(right_largest_contour, right_matrix, (cam_width, cam_height))

    # Display left and right webcam feeds with the contours displayed
    cv2.imshow('Left Webcam', left_largest_contour)
    cv2.imshow('Right Webcam', right_largest_contour)

    # SAVE IMAGE WHEN 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        data = {
            'Left Camera': {
                'Top Fisheye': {'X': int(left_largest[0][0][0]), 'Y': int(left_largest[0][0][1])},
                'Top Right': {'X': int(left_largest[1][0][0]), 'Y': int(left_largest[1][0][1])},
                'Bottom Fisheye': {'X': int(left_largest[2][0][0]), 'Y': int(left_largest[2][0][1])},
                'Bottom Right': {'X': int(left_largest[3][0][0]), 'Y': int(left_largest[3][0][1])},
            },
            'Right Camera': {
                'Top Fisheye': {'X': int(right_largest[0][0][0]), 'Y': int(right_largest[0][0][1])},
                'Top Right': {'X': int(right_largest[1][0][0]), 'Y': int(right_largest[1][0][1])},
                'Bottom Fisheye': {'X': int(right_largest[2][0][0]), 'Y': int(right_largest[2][0][1])},
                'Bottom Right': {'X': int(right_largest[3][0][0]), 'Y': int(right_largest[3][0][1])},
            }
        }

        # Writing to Corners.yaml
        with open("dist/RUNTIME DATA/Resources/Corners.yaml", "w") as outfile:
            yaml.dump(data, outfile, sort_keys=False)

        # Confirm saved parameters
        print("Calibration Data Saved!")

        # Exit program
        break
