import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def sliding_window(warped, warped_original, left_line, right_line, plot = False):
    if (left_line.detected == False) or (right_line.detected == False):
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
        left_line.appendNewFit(left_fit)
        right_line.appendNewFit(right_fit)
        left_line.getAverageFit()
        right_line.getAverageFit()
        left_line.detected = True
        right_line.detected = True

        curvature = get_curvature(lefty, leftx, righty, rightx)
        car_center = getCarCenter(left_line, right_line, warped)
        if plot:
            plot_slide_initial(warped, out_img, left_line.current_fit, right_line.current_fit, nonzeroy, nonzerox, left_lane_inds, right_lane_inds)
            return None
        return drawOnVideo(warped, warped_original, left_line, right_line, nonzeroy, nonzerox, left_lane_inds, right_lane_inds), curvature, car_center


    else:
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_line.current_fit[0]*(nonzeroy**2) + left_line.current_fit[1]*nonzeroy + left_line.current_fit[2] - margin)) & (nonzerox < (left_line.current_fit[0]*(nonzeroy**2) + left_line.current_fit[1]*nonzeroy + left_line.current_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_line.current_fit[0]*(nonzeroy**2) + right_line.current_fit[1]*nonzeroy + right_line.current_fit[2] - margin)) & (nonzerox < (right_line.current_fit[0]*(nonzeroy**2) + right_line.current_fit[1]*nonzeroy + right_line.current_fit[2] + margin)))
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        left_line.appendNewFit(left_fit)
        right_line.appendNewFit(right_fit)
        left_line.getAverageFit()
        right_line.getAverageFit()

        if plot:
            plot_slide(warped, left_line.current_fit, right_line.current_fit, nonzeroy, nonzerox, left_lane_inds, right_lane_inds, margin)
            return None
        curvature = get_curvature(lefty, leftx, righty, rightx)
        car_center = getCarCenter(left_line, right_line, warped)
        return drawOnVideo(warped, warped_original, left_line, right_line, nonzeroy, nonzerox, left_lane_inds, right_lane_inds), curvature, car_center

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

def get_curvature(lefty, leftx, righty, rightx):
    ym_per_pix = 30/720
    xm_per_pix = 3.7/700
    y_eval = np.max(lefty)
    #Fit polynomials in world space
    left_fit = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    #Calculate radii of curvature
    left_curve_rad = ((1 + (2 * left_fit[0] * y_eval * ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2 * left_fit[0])
    right_curve_rad = ((1 + (2 * right_fit[0] * y_eval * ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2 * right_fit[0])
    curvature = (left_curve_rad + right_curve_rad)/2
    return curvature

def drawOnVideo(warped, warped_original, left_line, right_line, nonzeroy, nonzerox, left_lane_inds, right_lane_inds):
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_line.current_fit[0]*ploty**2 + left_line.current_fit[1]*ploty + left_line.current_fit[2]
    right_fitx = right_line.current_fit[0]*ploty**2 + right_line.current_fit[1]*ploty + right_line.current_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = warped_original
    window_img = np.zeros_like(out_img)

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    window = np.hstack((left_line_window1, right_line_window2))
    cv2.fillPoly(window_img, np.int_([window]), (0,255, 0))
    return window_img

def getCarCenter(left_line, right_line, warped):
    xm_per_pix = 3.7/700
    l_base = left_line.current_fit[0]*warped.shape[0]**2 + left_line.current_fit[1]*warped.shape[0] + left_line.current_fit[2]
    r_base = right_line.current_fit[0]*warped.shape[0]**2 + right_line.current_fit[1]*warped.shape[0] + right_line.current_fit[2]
    car_center = (warped.shape[1]/2 - (l_base + r_base)/2)*xm_per_pix
    return car_center
