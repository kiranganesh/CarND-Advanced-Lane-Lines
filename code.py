import numpy as np
from ipywidgets import interact, interactive, fixed

import matplotlib.pyplot as plt

# -- Calculate Camera Distortion --

import cv2
import glob

# create subplot matrix of 20 images to plot
fig, ax = plt.subplots(5,4, figsize=(15,12))
ax = ax.ravel()

# datastructures for object & image points
objpoints = []
imgpoints = []

# create object points data based on expected chessboard pattern
pattern = np.zeros((6*9, 3), np.float32)
pattern[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

images = glob.glob('camera_cal/calibration*.jpg')

for i, filename in enumerate(images):

    # read the image 
    image = cv2.imread(filename)

    # convert to grayscale
    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # find the chessboard corners
    val, corners = cv2.findChessboardCorners(grayimage, (9,6), None)
    
    if val:

        # add to the list of image points        
        imgpoints.append(corners)
        objpoints.append(pattern)
                
        # display it on the image   
        cv2.drawChessboardCorners(image, (9,6), corners, val)

        # display the image
        ax[i].imshow(image)
    
plt.show()

# -- Test camerera image undistortion

img = cv2.imread('camera_cal/calibration1.jpg')
imgsize = (img.shape[1], img.shape[0])

images = glob.glob('camera_cal/calibration*.jpg')

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgsize,None,None)

fig2, ax2 = plt.subplots(5,4, figsize=(15,12))
ax2 = ax2.ravel()

for i, filename in enumerate(images):
   
    # read the image 
    image = cv2.imread(filename)

    # undistort the image 
    undist_image = cv2.undistort(image, mtx, dist, None, mtx)
    
    # display the image
    ax2[i].imshow(undist_image)

plt.show()

# -- Undistort a sample test image --

test1 = cv2.imread('test_images/test1.jpg')
test1 = cv2.cvtColor(test1, cv2.COLOR_BGR2RGB)
ud_test1 = cv2.undistort(test1, mtx, dist, None, mtx)

plt.imshow(test1)
plt.show()

plt.imshow(ud_test1)
plt.show()

# -- Perspective Transform

def perspective_transform(img, src, dst):
    
    h = img.shape[0]
    w = img.shape[1]

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)

    return warped

# -- Visualize an example

h = ud_test1.shape[0]
w = ud_test1.shape[1]

src = np.float32([(600, 450), (725,  450), (275, 675), (1100, 675)])
dst = np.float32([(400,   0), (800,    0), (400,   h), (800,    h)])

wp_test1 = perspective_transform(ud_test1, src, dst)

plt.imshow(wp_test1)
plt.show()

# -- Sobel Gradients --

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply x or y gradient with the OpenCV Sobel() function
    abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, orient == 'x', orient=='y'))

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)

    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def update(min_thresh, max_thresh):

    sobel_abs= abs_sobel_thresh(wp_test1, 'x', min_thresh, max_thresh)
    plt.imshow(sobel_abs, cmap='gray')

#interact(update, min_thresh=(0,255), max_thresh=(0,255))

sobel_abs= abs_sobel_thresh(wp_test1, 'x', 50, 250)
plt.imshow(sobel_abs, cmap='gray')





