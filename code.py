import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import ntpath

# Calibrate camera for distortion 

def calibrate_camera(nx, ny):
    
    print('CALIBRATING IMAGES')
    SHOW_IMAGES = False

    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    objpoints = []
    imgpoints = []

    # Make a list of calibration images
    images = glob.glob('camera_cal/*.jpg')

    # Step through the list and search for chessboard corners
    for i, fname in enumerate(images):

        print("Processing Image ", i)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            if SHOW_IMAGES:
                cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
                plt.imshow(img)
                plt.show()
    
    return (objpoints, imgpoints)

# Undistort an image based on calibrated camera

def undistort(img, objpoints, imgpoints):

    # img1 = cv2.imread('camera_cal/calibration1.jpg')
    # img2 = cv2.undistort(img1, mtx, dist, None, mtx)
    # plt.plot(img2)

    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    img_ud = cv2.undistort(img, mtx, dist, None, mtx)
    return img_ud 

# -- Apply Sobel and Color Processing to create binary image

def create_binary_image(img):

    # Sobel Transformation
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    retval, sxthresh = cv2.threshold(scaled_sobel, 30, 160, cv2.THRESH_BINARY)
    sxbinary[(sxthresh >= 30) & (sxthresh <= 160)] = 1

    # Color Transformation
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2] # get the S channel
    s_binary = np.zeros_like(s_channel) # initialize binary version
    s_thresh = cv2.inRange(s_channel.astype('uint8'), 170, 250) # sets to 255 if in range
    s_binary[(s_thresh == 255)] = 1 # normalize to 1 

    combined = np.zeros_like(gray)
    combined[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined

def perspective_transform(img):
    
    # Note: Image shape is 720 x 1280
    
    src = np.float32([[240,720],[580,450],[710,450],[1160,720]])
    dst =  np.float32([[300,720],[300,0],[900,0],[900,720]])

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

def inverse_perspective_transform(img):

    # Note: Image shape is 720 x 1280

    src = np.float32([[240,720],[580,450],[710,450],[1160,720]])
    dst =  np.float32([[300,720],[300,0],[900,0],[900,720]])

    M = cv2.getPerspectiveTransform(dst, src)
    return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

## -- Find Lanes from the perepective adjusted image 
def find_lanes(img):

    left_fit , right_fit =[], []
    out_img = np.dstack((img, img, img))*255

    histogram = np.sum(img[int(img.shape[0]/2):,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_size = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_size
        win_y_high = img.shape[0] - window*window_size
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # print(win_xleft_low, win_y_low, win_xleft_high, win_y_high)
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    fity = np.linspace(0, img.shape[0]-1, img.shape[0] )
    fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
    fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return (fit_leftx, fit_rightx, fity, out_img)

def fit_lane(warped_img, undist, yvals, left_fitx, right_fitx):

    # Create an image to draw the lines on
    color_warp = np.dstack((warped_img, warped_img, warped_img))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = inverse_perspective_transform(color_warp);
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

def get_curvature(leftx, rightx, ploty):

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

def draw_stat(img, left_curvature, right_curvature, distance):

    direction = 'L' if distance < 0 else 'R'
    cv2.putText(img, 'Left Curvature Radius = %d(m)' % left_curvature, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, 'Right Curvature Radius = %d(m)' % right_curvature, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, 'Distance from center = %d(cm) %s' % (np.absolute(distance), direction), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img;

# --- MAIN PROGRAM ---

objpoints, imgpoints = calibrate_camera(9,6)

def process_image(img):

    undistort_image = undistort(img, objpoints, imgpoints)
    processed_image = create_binary_image(undistort_image)
    processed_image = perspective_transform(processed_image)

    left_fit, right_fit, yvals, out_img = find_lanes(processed_image)
    processed_image = fit_lane(processed_image, undistort_image, yvals, left_fit, right_fit)
    left_curvature, right_curvature, distance = get_curvature(left_fit, right_fit, yvals)
    processed_image = draw_stat(processed_image, left_curvature, right_curvature, distance)

    return processed_image

def find_lanes_image():
    
    images = glob.glob('test_images/test*.jpg')

    for idx, fname in enumerate(images):
        image = cv2.imread(fname)
        processed_image = process_image(image)
        output_filename = 'output_images/' + ntpath.basename(fname)
        cv2.imwrite(output_filename, processed_image)
        plt.imshow(processed_image)
        plt.show()
        
def find_lanes_video():
    
    from moviepy.editor import VideoFileClip
    clip = VideoFileClip("project_video.mp4")

    output_video = "output_video/project_video.mp4"
    output_clip = clip.fl_image(process_image)
    output_clip.write_videofile(output_video, audio=False)

if __name__ == "__main__":

    # find_lanes_image()    
    find_lanes_video()
    
    
    
    
    