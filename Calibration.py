import numpy as np
import cv2

# Define the size of the checkerboard
CHECKERBOARD_SIZE = (8, 6)

# Define the termination criteria for the iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Define the object points for the calibration pattern
objp = np.zeros((np.prod(CHECKERBOARD_SIZE), 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)

# Define the arrays to store the calibration data for each camera
obj_points_l = []    # Object points in left camera
img_points_l = []    # Image points in left camera
obj_points_r = []    # Object points in right camera
img_points_r = []    # Image points in right camera

gray_l = None
gray_r = None

# Define the number of calibration images to use
NUM_IMAGES = 20

# Loop over the calibration images for each camera
for i in range(NUM_IMAGES):

    # Load the left and right images
    img_l = cv2.imread('Calibration Images/Left/left{}.jpg'.format(i))
    img_r = cv2.imread('Calibration Images/Right/right{}.jpg'.format(i))

    # Convert the images to grayscale
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # Find the corners of the calibration pattern in the images
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD_SIZE, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD_SIZE, None)

    # If the corners are found, add them to the calibration data
    if ret_l and ret_r:
        obj_points_l.append(objp)
        obj_points_r.append(objp)
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
        img_points_l.append(corners_l)
        img_points_r.append(corners_r)

# Calibrate the left camera
ret_l, mtx_l, dist_l, r_vectors_l, t_vectors_l =\
    cv2.calibrateCamera(obj_points_l, img_points_l, gray_l.shape[::-1], None, None)

# Calibrate the right camera
ret_r, mtx_r, dist_r, r_vectors_r, t_vectors_r =\
    cv2.calibrateCamera(obj_points_r, img_points_r, gray_r.shape[::-1], None, None)

# Calibrate the stereo system
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC    # Use the intrinsic parameters from the individual camera calibrations
stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F =\
    cv2.stereoCalibrate(obj_points_l, img_points_l, img_points_r, mtx_l, dist_l,
                        mtx_r, dist_r, gray_l.shape[::-1], criteria=stereo_criteria, flags=flags)

# Compute the rectification parameters for each camera
R_l, R_r, P_l, P_r, Q, valid_pix_roi_l, valid_pix_roi_r =\
    cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, gray_l.shape[::-1],
                      R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=1.0)

# Compute the un-distortion and rectification maps for each camera
map_x_l, map_y_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R_l, P_l, gray_l.shape[::-1], cv2.CV_32FC1)
map_x_r, map_y_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R_r, P_r, gray_r.shape[::-1], cv2.CV_32FC1)

# Define the image filenames for the left and right cameras
img_l_filename = 'Calibration Images/Stereo/left.jpg'
img_r_filename = 'Calibration Images/Stereo/right.jpg'

# Load the left and right images
img_l = cv2.imread(img_l_filename)
img_r = cv2.imread(img_r_filename)

# Un-distort and rectify the images
img_l_undistorted = cv2.remap(img_l, map_x_l, map_y_l, cv2.INTER_LINEAR)
img_r_undistorted = cv2.remap(img_r, map_x_r, map_y_r, cv2.INTER_LINEAR)

# Convert the undistorted and rectified images to grayscale
gray_l = cv2.cvtColor(img_l_undistorted, cv2.COLOR_BGR2GRAY)
gray_r = cv2.cvtColor(img_r_undistorted, cv2.COLOR_BGR2GRAY)

# Find the key points and descriptors in the left and right images
orb = cv2.ORB_create()
kp_l, des_l = orb.detectAndCompute(gray_l, None)
kp_r, des_r = orb.detectAndCompute(gray_r, None)

# Match the key points in the left and right images
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_l, des_r)

# Filter the matches using the Lowe's ratio test
matches = [m for m in matches if m.distance < 0.7 * matches[0].distance]

# Compute the disparity map from the filtered matches
disparity_map = np.full_like(gray_l, 255)
for m in matches:
    u_l, v_l = kp_l[m.queryIdx].pt
    u_r, v_r = kp_r[m.trainIdx].pt
    disparity_map[int(v_l), int(u_l)] = np.abs(u_l - u_r)

# Normalize the disparity map for display
disparity_map_norm = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

# Display the results
cv2.imshow('Left Image', img_l_undistorted)
cv2.imshow('Right Image', img_r_undistorted)
cv2.imshow('Disparity Map', disparity_map_norm)
cv2.waitKey(0)
cv2.destroyAllWindows()
