## Advanced Lane Finding

This repository contains the project I implemented as part of Udacity Self Driving Car Nano Degree Program.

The high level goal of this project is to implement an image pipeline to detect the lane lines from the given image or video.

### High level steps of this project are following,

* Compute camera calibration and distortion coefficients from the chessboard images.
* Apply distortion correction on the images/frames.
* Explore and find appropriate color transforms and gradient to create thresholded binary images
* Apply perspective transform to warp the image to create warped image.
* Identify the lane lines pixels and fit them to a second order polynomial.
* Compute the radius of the carvature of the lane and position of the vehicle from the center of lane.
* Perform inverse transform to plot the detected lanes back onto original road images. 

## Rubric Points 

In this section I will describe how each of the Rubric points are implemented in my implementation.

### Camera Calibration 

A stand-alone python script "calibrate_camera.py" was implemented to compute the camera calibration and distortion coeficients of the camera based on the given chessboard images. This python script uses openCV funtions (findChessboardCorners() and calibrateCamera()) to compute the camera calibartion as described in the class. OpenCV function undistort() can then be used to remove the camera distortion. This script also writes the calibration/coeficients to a file using pickle ("calibration.p"), so that this can be utilized in the image pipeline implementation.

Image below shows the result of distortion correction using camera calibration and disctortion coeficients on a eample chess board image.
![alt text](output_images/undistorted_chessboard.jpeg "Undistorted Chessboard image")

Following is another example of distortion corrected raod image.
![alt text](output_images/undistorted_test1.jpeg "Undistorted test road image" )






[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `output_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

