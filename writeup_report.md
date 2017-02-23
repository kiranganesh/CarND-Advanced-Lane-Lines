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

##Camera Calibration

The code for calibrating the camera is contained in two functions `calibrate_camera()` and `undistort_image()`. 

The `calibrate_camera()` function takes in the set of test images and uses the cv2.findChessboardCorners function to collect the image points. This, together with a known definition of objectpoints together enable the `undistort_image()` to calculate the distortion coefficients of the camera.

The sample undistorted image is given below:

![Image](https://github.com/kiranganesh/CarND-Advanced-Lane-Lines/blob/master/examples/image1.JPG)

##Processing Pipeline

The core pipeline is defined as follows:

def process_image(img):

    # Preprocess the image
    undistort_image = undistort(img, objpoints, imgpoints)
    processed_image = create_binary_image(undistort_image)
    processed_image = perspective_transform(processed_image)

    # Extract data from the image
    left_fit, right_fit, yvals, out_img = find_lanes(processed_image)
    processed_image = fit_lane(processed_image, undistort_image, yvals, left_fit, right_fit)
    left_curvature, right_curvature, distance = get_curvature(left_fit, right_fit, yvals)
    processed_image = draw_stat(processed_image, left_curvature, right_curvature, distance)
    return processed_image

The code for generating the binary image is contained in the `create_binary_image()` function. It captures both the thresholded Sobel gradient transformation as well as color threshold transformation.

The sample pictures of the individual and combined transformations are given below:

![Image](https://github.com/kiranganesh/CarND-Advanced-Lane-Lines/blob/master/examples/image2.JPG)


