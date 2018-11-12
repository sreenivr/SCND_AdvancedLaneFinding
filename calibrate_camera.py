# This is a standalone python script to compute the
# camera calibration matrix and distortion coefficients
# from the given chessboard images.
# It also tests/demostrates the undistortion on one of the
# calibration images.

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

nx = 9 #TODO: enter the number of inside corners in x
ny = 6 #TODO: enter the number of inside corners in y

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

cal_file_names = glob.glob('camera_cal/calibration*.jpg')

# Iterate through the all the calibration chessboard 
# images and search for corners.
for fname in cal_file_names:    
    # Read the image
    img = mpimg.imread(fname)
    
    # Convert the image to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
    # If corners are found, add objectpoints and imgpoints.
    if ret == True:
        print("Found coreners on", fname)
        imgpoints.append(corners)
        objpoints.append(objp)
        
        # Test code 
        img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        #plt.imshow(img)
        #plt.show()
    else:
        print("Error: Didn't find corners on", fname)

# Read one of the calibration images. We will use this 
# to find the size (width x height) of the image 
# as well as to test the camera calibration.
test_img = cv2.imread('camera_cal/calibration1.jpg')
img_size = (test_img.shape[1], img.shape[0])

# Calibrate camera        
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# Undistort an image
undistorted_img = cv2.undistort(test_img, mtx, dist, None, mtx)

# Save calibration values to a file.
camera_cal_values = {}
camera_cal_values['mtx'] = mtx
camera_cal_values['dist'] = dist
pickle.dump(camera_cal_values, open("calibration.p", "wb"))

# Display original image and undistorted image
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.05)
ax1.imshow(test_img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(undistorted_img)
ax2.set_title('Undistorted Image', fontsize=30)
plt.show()

        

