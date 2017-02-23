##Advanced Lane Finding Project


The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/image1.jpg "Undistorted"
[video1]: ./project_video.mp4 "Video"

##Camera Calibration

The code for calibrating the camera is contained in two functions `calibrate_camera()` and `undistort_image()`. The former function takes in the set of test images and uses the cv2.findChessboardCorners function to collect the image points. This, together with a known definition of objectpoints together enable opencv API to calculate the distortion coefficients of the camera.

![alt text][image1]
