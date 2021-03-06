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

The perspective transform was calculated through a manual trial and error process by cutting out a trapezoid in the bottom half of the image. This results in a transformation of the image like this:

![Image](https://github.com/kiranganesh/CarND-Advanced-Lane-Lines/blob/master/examples/image4.JPG)

##Step 4. Find Lanes

The `find_lanes()`procedure determines the left and right edges of the lane from the perspective transformed image. 

The output of this stage looks like this, with the original image included for comparison.

![Image](https://github.com/kiranganesh/CarND-Advanced-Lane-Lines/blob/master/examples/image5.JPG)

The `fit_lanes()` procedure puts a filled polygon that is warped back to the original image.

![Image](https://github.com/kiranganesh/CarND-Advanced-Lane-Lines/blob/master/examples/image7.JPG)

Both these functions are largely based on the example code that was already provided by Udacity as part of the lessons.

##Step 5. Determine Curvature

Again, with the assistance of the code already provided by Udacity, the curvature of the left and right lanes is computed as well as the distance by which the vehicle is off center.

    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/590 # meters per pixel in x dimension
    y_eval = np.max(ploty)

    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Write dist from center
    center = 640.
    lane_x = rightx - leftx
    center_x = (lane_x / 2.0) + leftx
    distance = (center_x - center) * xm_per_pix

    return (left_curverad, right_curverad, np.mean(distance * 100.0))

##Step 6. Annotate the image 

Once the curvature values are known, the text data is added to the pictures as an annotation. 

    direction = 'L' if distance < 0 else 'R'
    cv2.putText(img, 'Left Curvature Radius = %d(m)' % left_curvature, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, 'Right Curvature Radius = %d(m)' % right_curvature, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, 'Distance from center = %d(cm) %s' % (np.absolute(distance), direction), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
##Final Output

The main function of the program has two routines 

    find_lanes_image()    
    find_lanes_video()

The first one processes the test images and the second one process the video.

A sample image that is processed through the full pipeline looks like this:

![Image](https://github.com/kiranganesh/CarND-Advanced-Lane-Lines/blob/master/examples/image6.JPG)

A sample video that is processed through the full pipeline looks like this:

https://github.com/kiranganesh/CarND-Advanced-Lane-Lines/blob/master/output_video/project_video.mp4

##Comments 

The main learning for me from this project was the understanding of the various steps to go from a raw highway image to a higher level mathematical representation of a lane and the vehicle's position in it. What seems like an abstract challenge is broken down into a series of simple and manageable steps.

Finding the right threshold values for the binary images was a difficult and time intensive process. I am not sure if the solution I have is necessarily an optimal one; its just a working solution. An improvement for the future would be to write some code to automatically find an optimal threshold instead of experiementing manually. The perspective transform also was done with manual geometries and possibly there's a better way to do it more automatically.

All of these seems to work because there is no traffic in the lane of interest. Looking at the video of the harder challenge, I can at once see several complications - varying degrees of sunlight against the camera, vehicle that cuts in on the front, non uniform terrain of the road etc. That challenge video very quickly makes it clear that this example here is just a simple start and a lot more considerations need to be addressed before it becomes a more robust solution for real list.











