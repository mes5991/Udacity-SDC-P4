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
    #get x gradient
    gradx = abs_sobel_thresh(img, orient='x', thresh=(20,150))
    #get binary image from saturation threshold
    sbinary = threshold_s_channel(img, thresh=(150,255))[0]
    #combine binary images with OR
    combined_binary = np.zeros_like(sbinary)
    combined_binary[(gradx == 1) | (sbinary == 1)] = 1
    return combined_binary

def test_roi(img):
    masked = region_of_interest(img)
    img_size = (img.shape[1], img.shape[0])
    vertices = np.array([[((img_size[0] / 2) - 65, img_size[1] / 2 + 50), \
                      (((img_size[0] / 6) - 50), img_size[1]), \
                      ((img_size[0] * 5 / 6) + 200, img_size[1]), \
                      ((img_size[0] / 2 + 65), img_size[1] / 2 + 50)]], dtype=np.int32)

    img = cv2.line(img, (vertices[0][0][0], vertices[0][0][1]), (vertices[0][1][0], vertices[0][1][1]), color=(255,0,0), thickness=2)
    img = cv2.line(img, (vertices[0][1][0], vertices[0][1][1]), (vertices[0][2][0], vertices[0][2][1]), color=(255,0,0), thickness=2)
    img = cv2.line(img, (vertices[0][2][0], vertices[0][2][1]), (vertices[0][3][0], vertices[0][3][1]), color=(255,0,0), thickness=2)
    img = cv2.line(img, (vertices[0][3][0], vertices[0][3][1]), (vertices[0][0][0], vertices[0][0][1]), color=(255,0,0), thickness=2)
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.imshow(masked)
    plt.show()

def region_of_interest(img):
    img_size = (img.shape[1], img.shape[0])
    vertices = np.array([[((img_size[0] / 2) - 65, img_size[1] / 2 + 50), \
                      (0, img_size[1]), \
                      (img_size[0], img_size[1]), \
                      ((img_size[0] / 2 + 65), img_size[1] / 2 + 50)]], dtype=np.int32)
    #defining a blank mask
    mask = np.zeros_like(img)

    ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def normalize(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=.2, tileGridSize=(3,3))
    yuv[:,:,0] = clahe.apply(yuv[:,:,0])
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    return rgb

def sliding_window(warped, left_fit, right_fit):
    if left_fit == None:
        # Take a histogram of the bottom half of the image
        histogram = np.sum(warped[warped.shape[0]/2:,:], axis=0)

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((warped, warped, warped))*255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 10
        # Set height of windows
        window_height = np.int(warped.shape[0]/nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Set the width of the windows +/- margin
        margin = 50
        # Set minimum number of pixels found to recenter window
        minpix = 200
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = warped.shape[0] - (window+1)*window_height
            win_y_high = warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

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
        plot_slide_initial(warped, out_img, left_fit, right_fit, nonzeroy, nonzerox, left_lane_inds, right_lane_inds)
        return left_fit, right_fit

    else:
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        plot_slide(warped, left_fit, right_fit, nonzeroy, nonzerox, left_lane_inds, right_lane_inds, margin)
        return left_fit, right_fit

def plot_slide_initial(warped, out_img, left_fit, right_fit, nonzeroy, nonzerox, left_lane_inds, right_lane_inds):
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.pause(.01)
    plt.clf()

def plot_slide(warped, left_fit, right_fit, nonzeroy, nonzerox, left_lane_inds, right_lane_inds, margin):
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((warped, warped, warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.pause(.01)
    plt.clf()
