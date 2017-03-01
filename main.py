import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from calibration import *
from transform import *
from imageProcessing import *
from lineFinder import *
from Line import Line

options = sys.argv

#Calibrate camera
ret, mtx, dist, rvecs, tvecs = calibrate_camera(plot=False)
print("Camera Calibrated!")

if "make_video" in options:
    '''Runs thru entire image processing pipeline and writes final output video.'''

    vid_path = r'C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\project_video.mp4'
    #Open video reader
    cap = cv2.VideoCapture(vid_path)
    #Open video writer
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    videoout = cv2.VideoWriter('output.avi',fourcc, 30.0,( 1280,720))
    #Create left and right lane line objects
    left_line = Line()
    right_line = Line()
    i=0
    while(cap.isOpened()):
        #Read frame of video
        ret, frame = cap.read()
        i += 1
        #End while loop if video ends
        if not ret:
            break
        #Run frame thru image pipeline for undistorting, perspective transformation, and segmentation
        result, threshold_img, warped, undist, Minv = image_pipeline(frame, mtx, dist)
        #Fit polynomials to segmented image and return a mask of the space inbetween left and right lines
        green_space, curvature, car_center = sliding_window(result, warped, left_line, right_line, plot = False)
        #Warp mask back to video perspective
        green_space_transform = cv2.warpPerspective(green_space, Minv, (green_space.shape[1], green_space.shape[0]))
        #Combine original video frame and transformed mask of lane
        output = cv2.addWeighted(frame, 1, green_space_transform, 0.3, 0)
        #Display lane curvature and car distance from center on frame
        cv2.putText(output, "Lane Curvature: " + str(int(curvature)) + " Meters", (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness = 2)
        cv2.putText(output, "Car Distance From Center: " + "{0:.2f}".format(car_center), (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness = 2)
        #Write to output video
        videoout.write(output)
        if i % 100 == 0:
            print("Writing Frame:", i)
    cap.release()
    cv2.destroyAllWindows()

if 'play_video' in options:
    '''Runs thru entire image processing pipeline and displays final output video, but does not save the video.'''

    vid_path = r'C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\project_video.mp4'
    cap = cv2.VideoCapture(vid_path)
    left_line = Line()
    right_line = Line()
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        result, threshold_img, warped, undist, Minv = image_pipeline(frame, mtx, dist)
        green_space, curvature, car_center = sliding_window(result, warped, left_line, right_line, plot = False)
        green_space_transform = cv2.warpPerspective(green_space, Minv, (green_space.shape[1], green_space.shape[0]))
        output = cv2.addWeighted(frame, 1, green_space_transform, 0.3, 0)
        cv2.putText(output, "Lane Curvature: " + str(int(curvature)) + " Meters", (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness = 2)
        cv2.putText(output, "Car Distance From Center: " + "{0:.2f}".format(car_center), (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness = 2)
        cv2.imshow('output', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if 'play_video_transform' in options:
    '''Runs thru image processing pipeline but stops after polynomial fitting. Displays perspective-transformed video with fitted polynomials'''

    vid_path = r'C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\project_video.mp4'
    cap = cv2.VideoCapture(vid_path)
    left_line = Line()
    right_line = Line()
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        i += 1
        if not ret:
            break
        result, threshold_img, warped, undist, Minv = image_pipeline(frame, mtx, dist)
        plt.ion()
        ret = sliding_window(result, warped, left_line, right_line, plot = True)
    cap.release()
    cv2.destroyAllWindows()

if 'test_calibration' in options:
    '''Displays undistored checkerboard images using camera calibration computed above'''

    test_calibration(mtx, dist)
    print("Calibration Tested!")

if 'test_calibration_single_image' in options:
    '''Displays undistored test image using camera calibration computed above'''
    img_path = r'C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images\test1.jpg'
    test_calibration_single_image(mtx, dist, img_path)

if 'test_transform' in options:
    '''Displays transformed checkerboard images'''

    test_transform(r'C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images\\', mtx, dist)
    print("Transforms tested!")

if 'test_thresh' in options:
    '''Displays binary thresholding steps used in image processing'''

    test_path = r"C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images"
    test_imgs = os.listdir(test_path)
    for i in test_imgs:
        img_path = test_path + '/' + i
        img = mpimg.imread(img_path)
        undist = undistort(img, mtx, dist)
        undist = cv2.GaussianBlur(undist, (5,5), 0)
        undist = normalize(undist)
        warped, dst, src, Minv = transform(undist)

        combined_binary, gradx, sbinary, gray_binary, morphed, masked = thresh_pipeline(warped)

        plt.subplot(2,3,1)
        plt.imshow(warped)
        plt.title("warped")
        plt.subplot(2,3,2)
        plt.imshow(gradx, cmap='gray')
        plt.title("gradx")
        plt.subplot(2,3,3)
        plt.imshow(sbinary, cmap='gray')
        plt.title("sbinary")
        plt.subplot(2,3,4)
        plt.imshow(gray_binary, cmap='gray')
        plt.title("gray_binary")
        plt.subplot(2,3,5)
        plt.imshow(masked, cmap='gray')
        plt.title("gray_binary_masked")
        plt.subplot(2,3,6)
        plt.imshow(combined_binary, cmap='gray')
        plt.title("combined_binary")
        plt.show()

if 'test_histo' in options:
    '''Displays histogram on top of perspective-transformed and thresholded image'''

    test_path = r"C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images"
    test_imgs = os.listdir(test_path)
    for i in test_imgs:
        img_path = test_path + '/' + i
        img = mpimg.imread(img_path)
        # warped, undist, masked = image_pipeline(img)
        result, threshold_img, warped, undist, Minv = image_pipeline(img, mtx, dist)
        histogram = np.sum(result[result.shape[0]/2:,:], axis=0)
        plt.figure()
        plt.plot(result.shape[0] - histogram, color=(1,0,0))
        plt.imshow(result, cmap='gray')
        plt.title("Histogram")
        plt.show()
