import yaml
import numpy as np
import CalibrationUtilities as Utils
import cv2
from screeninfo import get_monitors
import vlc

# Burlington Central High School -- TEJ4M1 'The Log Project' --> 'Touch Screen Projector V2' By: Majd Aburas

# Define camera and projection dimensions in pixels
cam_width, cam_height = 1920, 1080

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

# Retrieve camera matrix and distortion coefficients for each camera
K = np.array(calib_data['camera_0']['K'])
D = np.array(calib_data['camera_0']['D'])

# Calculate the perspective transformation matrix to un-distort webcams' fisheye effect
fisheye_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K, D, (cam_width, cam_height), np.eye(3), balance=0.0)

dst = np.float32([[0, 0], [cam_width, 0], [0, cam_height], [cam_width, cam_height]])
mod = np.float32([[crop_offset, crop_offset], [cam_width - crop_offset, crop_offset],
                  [crop_offset, cam_height - crop_offset], [cam_width - crop_offset, cam_height - crop_offset]])

# Initialize trackbars for contour filtering
Utils.initialize_trackbars()

# Define the parameters of the concentric rings
x = 500
y = 500
num_rings = 5
radius_step = 50

# Create the rings on a blank image
img = np.zeros((1000, 1000, 3), np.uint8)
for i in range(num_rings):
    cv2.circle(img, (x, y), radius_step * (i+1), (255, 255, 255), 2)
# Define the media file to be played
media_file = "path/to/your/media/file.mp4"

# Create the VLC instance and media player
vlc_instance = vlc.Instance()
player = vlc_instance.media_player_new()

while True:
    # Read webcam feeds
    success, img = capture.read()

    if success:
        # Apply fisheye distortion removal to left and right separately
        img = cv2.fisheye.undistortImage(img, K, D, None, Knew=fisheye_matrix)
        img = cv2.resize(img, (1024, 576))

        # Retrieve trackbar values for thresholding
        threshold = Utils.val_trackbars()
        kernel = np.ones((5, 5))

        # Apply image filters to allow for contour filtering
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to Grayscale
        blur = cv2.GaussianBlur(gray, (5, 5), 1)  # Add Gaussian blur
        canny = cv2.Canny(blur, threshold[0], threshold[1])  # Apply Canny blur
        dilation = cv2.dilate(canny, kernel, iterations=2)  # Apply Dilation
        canny = cv2.erode(dilation, kernel, iterations=1)  # Apply Erosion

        # Find all contours in Image
        largest_contour = img.copy()  # Copy image for display purposes
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Isolate the largest contour by area
        largest, max_area = Utils.biggest_contour(contours)

        # Draw the largest contour found
        if largest.size != 0:
            largest = Utils.reorder(largest)
            cv2.drawContours(largest_contour, largest, -1, (0, 255, 0), 20)  # Draw the largest contour
            largest_contour = Utils.draw_rectangle(largest_contour, largest, 2)
            src = np.float32(largest)
            matrix = cv2.getPerspectiveTransform(src, dst)
            # largest_contour = cv2.warpPerspective(largest_contour, matrix, (cam_width, cam_height))

        # Display left and right webcam feeds with the contours displayed
        cv2.imshow('Undistorted Webcam Feed', largest_contour)

        # SAVE IMAGE WHEN 's' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('s'):
            if largest.size > 0:
                data = {
                    'Top Left': {'X': int(largest[0][0][0]), 'Y': int(largest[0][0][1])},
                    'Top Right': {'X': int(largest[1][0][0]), 'Y': int(largest[1][0][1])},
                    'Bottom Left': {'X': int(largest[2][0][0]), 'Y': int(largest[2][0][1])},
                    'Bottom Right': {'X': int(largest[3][0][0]), 'Y': int(largest[3][0][1])},
                }
            else:
                data = {
                    'Top Left': {'X': 0, 'Y': 0},
                    'Top Right': {'X': cam_width, 'Y': 0},
                    'Bottom Left': {'X': 0, 'Y': cam_height},
                    'Bottom Right': {'X': cam_width, 'Y': cam_height},
                }

            # Writing to Corners.yaml
            with open("dist/RUNTIME DATA/Resources/Corners.yaml", "w") as outfile:
                yaml.dump(data, outfile, sort_keys=False)

            # Confirm saved parameters
            print("Calibration Data Saved!")

            # Exit program
            break
    else:
        print('No camera feed(s) found, Please double check connections or restart device!')
