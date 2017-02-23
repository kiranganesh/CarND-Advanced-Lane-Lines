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

The core pipeline is defined as follows in the `process_image()` function.

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

Each of the above steps in the pipeline is described in more detail below.

##Step 1. Correcting for Distortion

The `undistort_image()` function described earlier is the first step of the pipeline and it corrects for the camera distortions

![Image](https://github.com/kiranganesh/CarND-Advanced-Lane-Lines/blob/master/examples/image3.JPG)

##Step 2. Creating Binary Image

The code for generating the binary image is contained in the `create_binary_image()` function. It captures both the thresholded Sobel gradient transformation as well as color threshold transformation.

The sample pictures of the individual and combined transformations are given below:

![Image](https://github.com/kiranganesh/CarND-Advanced-Lane-Lines/blob/master/examples/image2.JPG)

##Step 3. Perspective Transformation

Once the binary image is created, the following perspective transformation is used to create the birds-eye view of the portion of the road that's of interest. 

    src = np.float32([[240,720],[580,450],[710,450],[1160,720]])
    dst =  np.float32([[300,720],[300,0],[900,0],[900,720]])

    M = cv2.getPerspectiveTransform(src, dst)
    img2 = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

This results in a transformation of the image like this:

![Image](https://github.com/kiranganesh/CarND-Advanced-Lane-Lines/blob/master/examples/image4.JPG)



