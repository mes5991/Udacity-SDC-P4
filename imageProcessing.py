import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from transform import *

def image_pipeline(img, mtx, dist):
    #Undistort the image
    undist = undistort(img, mtx, dist)
    undist = cv2.GaussianBlur(undist, (5,5), 0)
    undist = normalize(undist)
    #Warp perspective
    warped, dst, src, Minv = transform(undist)
    #Threshold to generate binary image where lane lines are clear
    threshold_img = thresh_pipeline(warped)[0]
    #Use morphology to clean up the binary image
    kernel = np.ones((5,5),np.uint8)
    result = cv2.morphologyEx(threshold_img, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((10,10),np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return result, threshold_img, warped, undist, Minv

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

def test_sobel_thresh(img, ksize=3):
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20,150))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(20,100))
    mag_bin = mag_thresh(img, sobel_kernel=ksize, thresh=(30,100))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.7,1.2))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_bin == 1) & (dir_binary == 1))] = 1

    plt.figure(figsize=(15,8))
    plt.subplot(3,3,1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.subplot(3,3,2)
    plt.imshow(gradx, cmap='gray')
    plt.title("Gradx Image")
    plt.subplot(3,3,3)
    plt.imshow(grady, cmap='gray')
    plt.title("Grady Image")
    plt.subplot(3,3,4)
    plt.imshow(mag_bin, cmap='gray')
    plt.title("mag_bin Image")
    plt.subplot(3,3,5)
    plt.imshow(dir_binary, cmap='gray')
    plt.title("dir_binary Image")
    plt.subplot(3,3,6)
    plt.imshow(combined, cmap='gray')
    plt.title("Processed Image")
    plt.subplots_adjust(hspace=1)
    plt.show()

def threshold_s_channel(img, thresh=(90,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary, S

def threshold_white(img, thresh=(190,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    result = np.zeros_like(gray)
    thresh = 175
    result[gray > thresh] = 1
    while np.mean(result) > .02:
        thresh += 1
        result = np.zeros_like(gray)
        result[gray > thresh] = 1
    kernel = np.ones((25,25),np.uint8)
    morphed = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    masked = np.copy(result)
    masked[(result > 0) & (morphed > 0)] = 0
    return result, morphed, masked

def test_color_thresholding(img, thresh):
    thresh_S, S = threshold_s_channel(img, thresh=thresh)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_thresh = (200, 255)
    binary = np.zeros_like(gray)
    binary[(gray > gray_thresh[0]) & (gray <= gray_thresh[1])] = 1

    plt.subplot(2,2,1)
    plt.imshow(img)
    plt.subplot(2,2,2)
    plt.imshow(S, cmap='gray')
    plt.subplot(2,2,3)
    plt.imshow(binary, cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(thresh_S, cmap='gray')
    plt.show()

def thresh_pipeline(img):
    #get x gradient
    gradx = abs_sobel_thresh(img, orient='x', thresh=(20,150))
    #get binary image from saturation threshold
    sbinary = threshold_s_channel(img, thresh=(130,255))[0]
    #get binary image using grayscale
    gray_binary, morphed, masked = threshold_white(img)
    #combine binary images
    combined_binary = np.zeros_like(sbinary)
    combined_binary[(gradx == 1) | (sbinary == 1) | (gray_binary == 1) | (masked == 1)] = 1
    return combined_binary, gradx, sbinary, gray_binary, morphed, masked


def normalize(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=.2, tileGridSize=(3,3))
    yuv[:,:,0] = clahe.apply(yuv[:,:,0])
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    return rgb
