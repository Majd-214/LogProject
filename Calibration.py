import numpy as np
import cv2
import yaml

# Initialize the calibration data dictionary
calibration_data = {}

# Define the chessboard pattern size
pattern_size = (9, 6)

# Define the number of calibration images to use
num_images = 20

# Define the size of the pattern's squares in units (cm)
square_size = 2.2

# Define the camera indexes chosen
left_camera_index, right_camera_index = 0, 1

# Define the world coordinates of the chessboard corners
world_coordinates = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
world_coordinates[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# Define the termination criteria for the iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# Function to calibrate a single camera
def calibrate_camera(camera_index):
    # Create lists to store object points and image points from all images
    obj_points = []
    img_points = []

    # Initialize the grayscale image variable
    gray = None

    # Loop over all calibration images and find chessboard corners
    for i in range(num_images):
        # Load the calibration image
        img = cv2.imread('dist/RUNTIME DATA/Resources/Calibration Images/Fisheye/camera_{}_calib_{}.jpg'
                         .format(camera_index, i))

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret_c, corners = cv2.findChessboardCorners(gray, pattern_size)

        # If the corners are found, add the object points and image points to the lists
        if ret_c:
            obj_points.append(world_coordinates)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners)

        print('Iteration: ' + str(i) + ' --> Corners found: ' + str(ret_c))

    # Use the object points and image points to perform camera calibration
    ret_c, camera_matrix, dist_coefficients, r_vectors, t_vectors =\
        cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    # Store the calibration data in the dictionary
    calibration_data[f'camera_{camera_index}'] = {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coefficients': dist_coefficients.tolist()
    }


# Function to perform stereo calibration
def calibrate_stereo():
    # Load the chessboard images captured by each camera
    img_l = cv2.imread('dist/RUNTIME DATA/Resources/Calibration Images/Stereo/left.jpg')
    img_r = cv2.imread('dist/RUNTIME DATA/Resources/Calibration Images/Stereo/right.jpg')

    # Load the camera matrices and distortion coefficients computed in the calibrate_camera function
    camera_matrix_l = np.array(calibration_data[f'camera_{left_camera_index}']['camera_matrix'])
    dist_coefficients_l = np.array(calibration_data[f'camera_{left_camera_index}']['dist_coefficients'])
    camera_matrix_r = np.array(calibration_data[f'camera_{right_camera_index}']['camera_matrix'])
    dist_coefficients_r = np.array(calibration_data[f'camera_{right_camera_index}']['dist_coefficients'])

    # Un-distort the images using the calibration parameters
    img_l = cv2.undistort(img_l, camera_matrix_l, dist_coefficients_l)
    img_r = cv2.undistort(img_r, camera_matrix_r, dist_coefficients_r)

    # Find the chessboard corners in the images
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, pattern_size)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, pattern_size)

    # If the corners are found in both images, refine the corner positions and compute the stereo calibration
    if ret_l and ret_r:
        # Refine the corner positions for greater accuracy
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

        # Create a list of the object points (the 3D points of the chessboard corners)
        object_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        object_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        object_points *= square_size

        # Create a list of the corresponding image points for each camera
        image_points_l = corners_l.reshape(-1, 2)
        image_points_r = corners_r.reshape(-1, 2)

        # Perform stereo calibration to compute the rotation and
        # translation matrices and the essential and fundamental matrices
        _, camera_matrix_l, dist_coefficients_l, camera_matrix_r, dist_coefficients_r, r, t, e, f =\
            cv2.stereoCalibrate(
                [object_points], [image_points_l], [image_points_r],
                camera_matrix_l, dist_coefficients_l, camera_matrix_r, dist_coefficients_r,
                gray_l.shape[::-1], criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)

        # Compute the rectification transforms and projection matrices for each camera
        r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
            camera_matrix_l, dist_coefficients_l, camera_matrix_r, dist_coefficients_r,
            gray_l.shape[::-1], r, t, alpha=-1, newImageSize=gray_l.shape[::-1])

        # Store the stereo calibration data in the dictionary
        calibration_data['stereo'] = {
            'R': r.tolist(),
            'T': t.tolist(),
            'E': e.tolist(),
            'F': f.tolist(),
            'R1': r1.tolist(),
            'R2': r2.tolist(),
            'P1': p1.tolist(),
            'P2': p2.tolist(),
            'Q': q.tolist()
        }

        # Save the calibration data to a YAML file
        with open('dist/RUNTIME DATA/Resources/Calibration.yaml', 'w') as f:
            yaml.dump(calibration_data, f)


# Main function to run the calibration process
def main():
    # Call the calibrate_camera function for each camera
    calibrate_camera(left_camera_index)
    calibrate_camera(right_camera_index)

    # Call the calibrate_stereo function to perform stereo calibration
    calibrate_stereo()
    print("Successfully Calibrated!")


if __name__ == '__main__':
    main()
