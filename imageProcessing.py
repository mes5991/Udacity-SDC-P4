import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
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
    # gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20,150))
    # grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(20,100))
    # mag_bin = mag_thresh(img, sobel_kernel=ksize, thresh=(30,100))
    # dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.7,1.2))

    ksize=7
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(10,255))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(60,255))
    mag_bin = mag_thresh(img, sobel_kernel=ksize, thresh=(40,255))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(.65,1.05))

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
    return binary

def test_color_thresholding(img_path, thresh):
    img = mpimg.imread(img_path)
    thresh_S = threshold_s_channel(img, thresh=thresh)

    plt.imshow(thresh_S, cmap='gray')
    plt.show()

def test_thresh(img_path, S_thresh, gradx_thresh):
    img = mpimg.imread(img_path)
    gradx = abs_sobel_thresh(img, orient='x', thresh=gradx_thresh)
    sbinary = threshold_s_channel(img, thresh=S_thresh)

    color_binary = np.dstack((np.zeros_like(gradx), gradx, sbinary))
    print(gradx.shape)
    print(sbinary.shape)

    combined_binary = np.zeros_like(sbinary)
    combined_binary[(gradx == 1) | (sbinary == 1)] = 1

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.set_title("Stacked Thresholds")
    ax1.imshow(color_binary)
    ax2.set_title("Combined Thresholds")
    ax2.imshow(combined_binary, cmap='gray')
    plt.show()

def thresh_pipeline(img):
    # gradx = abs_sobel_thresh(img, orient='x', thresh=(20,150))

    ksize=7
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(10,255))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(60,255))
    mag_bin = mag_thresh(img, sobel_kernel=ksize, thresh=(40,255))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(.65,1.05))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_bin == 1) & (dir_binary == 1))] = 1

    sbinary = threshold_s_channel(img, thresh=(160,255))

    combined_binary = np.zeros_like(sbinary)
    # combined_binary[(gradx == 1) | (sbinary == 1)] = 1
    combined_binary[(gradx == 1) | (combined == 1)] = 1
    combined_binary = region_of_interest(combined_binary)
    return combined_binary

def test_roi(img):
    imshape = img.shape
    left_bottom = (100, imshape[0])
    right_bottom = (imshape[1]-20, imshape[0])
    apex1 = (610, 410)
    apex2 = (680, 410)
    inner_left_bottom = (310, imshape[0])
    inner_right_bottom = (1150, imshape[0])
    inner_apex1 = (700,480)
    inner_apex2 = (650,480)
    vertices = np.array([[left_bottom, apex1, apex2, \
                          right_bottom, inner_right_bottom, \
                          inner_apex1, inner_apex2, inner_left_bottom]], dtype=np.int32)

    masked = region_of_interest(img, vertices)

    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.imshow(masked)
    plt.show()


def region_of_interest(img, vertices):
    #defining a blank mask
    mask = np.zeros_like(img)

    ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
