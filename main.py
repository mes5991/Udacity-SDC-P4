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

options = sys.argv

#Calibrate camera
ret, mtx, dist, rvecs, tvecs = calibrate_camera(plot=False)
print("Camera Calibrated!")

def image_pipeline(img):
    #Undistort the image
    undist = undistort(img, mtx, dist)
    undist = cv2.GaussianBlur(undist, (5,5), 0)
    undist = normalize(undist)

    #Warp perspective
    warped, dst, src = transform1(undist)

    #Threshold to generate binary image where lane lines are clear
    threshold_img = thresh_pipeline(warped)

    #Mask image to region of interest
    # masked = region_of_interest(threshold_img)

    kernel = np.ones((9,9),np.uint8)
    result = cv2.morphologyEx(threshold_img, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((10,10),np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    return result, threshold_img, warped, undist

#Testing below
if 'test_calibration' in options:
    test_calibration(mtx, dist)
    # test_calibration_single_image(mtx, dist, r"C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images\straight_lines1.jpg")
    # test_calibration_single_image(mtx, dist, r"C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images\straight_lines2.jpg")
    print("Calibration Tested!")

if 'test_transform' in options:
    test_transform(r'C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images\\', mtx, dist)
    print("Transforms tested!")

if 'test_sobel_thresh' in options:
    test_path = r"C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images"
    test_imgs = os.listdir(test_path)
    for i in test_imgs:
        img_path = test_path + '/' + i
        img = mpimg.imread(img_path)
        undist = undistort(img, mtx, dist)
        undist = cv2.GaussianBlur(undist, (5,5), 0)
        test_sobel_thresh(undist)

if 'test_color_thresh' in options:
    test_path = r"C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images"
    test_imgs = os.listdir(test_path)
    for i in test_imgs:
        img_path = test_path + '/' + i
        img = mpimg.imread(img_path)
        undist = undistort(img, mtx, dist)
        undist = cv2.GaussianBlur(undist, (5,5), 0)
        test_color_thresholding(undist, (90,255))

if 'test_thresh' in options:
    test_path = r"C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images"
    test_imgs = os.listdir(test_path)
    for i in test_imgs:
        img_path = test_path + '/' + i
        test_thresh(img_path, S_thresh=(160,255), gradx_thresh=(20,150))

if 'images_hls' in options:
    test_path = r"C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images"
    test_imgs = os.listdir(test_path)
    for i in test_imgs:
        img_path = test_path + '/' + i
        img = mpimg.imread(img_path)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        H = hls[:,:,0]
        L = hls[:,:,1]
        S = hls[:,:,2]
        plt.figure(figsize=(20,10))
        plt.subplot(2,2,1)
        plt.imshow(img)
        plt.subplot(2,2,2)
        plt.imshow(H)
        plt.subplot(2,2,3)
        plt.imshow(L)
        plt.subplot(2,2,4)
        plt.imshow(S)
        plt.show()

if "test_roi" in options:
    test_path = r"C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images"
    test_imgs = os.listdir(test_path)
    for i in test_imgs:
        img_path = test_path + '/' + i
        img = mpimg.imread(img_path)
        test_roi(img)

if 'temp' in options:
    test_path = r"C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images"
    test_imgs = os.listdir(test_path)
    for i in test_imgs:
        img_path = test_path + '/' + i
        img = mpimg.imread(img_path)
        blur = cv2.GaussianBlur(img, (5,5), 0)
        plt.figure(figsize=(20,10))
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.subplot(1,2,2)
        plt.imshow(blur)
        plt.show()

if 'test_pipeline' in options:
    test_path = r"C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images"
    test_imgs = os.listdir(test_path)
    for i in test_imgs:
        img_path = test_path + '/' + i
        img = mpimg.imread(img_path)
        result, threshold_img, warped, undist = image_pipeline(img)

        plt.figure(figsize=(20,10))
        plt.subplot(2,2,1)
        plt.imshow(img)
        plt.subplot(2,2,2)
        plt.imshow(undist)
        plt.subplot(2,2,3)
        plt.imshow(threshold_img, cmap='gray')
        plt.subplot(2,2,4)
        plt.imshow(result, cmap='gray')
        plt.show()

if 'test_video' in options:
    vid_path = r'C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\project_video.mp4'
    # vid_path = r'C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\harder_challenge_video.mp4'
    cap = cv2.VideoCapture(vid_path)
    left_fit = None
    right_fit = None
    while(cap.isOpened()):
        ret, frame = cap.read()
        warped, undist, masked = image_pipeline(frame)
        histogram = warped.shape[0] - np.sum(warped[warped.shape[0]/2:,:], axis=0) - 1
        warped[warped > 0] = 255
        masked[masked > 0] = 255
        # warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        plt.ion()
        left_fit, right_fit = sliding_window(warped, left_fit, right_fit)

        # cv2.imshow('Video', frame)
        # cv2.imshow('warped', warped)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()

if 'test_histo' in options:
    test_path = r"C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images"
    test_imgs = os.listdir(test_path)
    for i in test_imgs:
        img_path = test_path + '/' + i
        img = mpimg.imread(img_path)
        warped, undist, masked = image_pipeline(img)
        histogram = np.sum(warped[warped.shape[0]/2:,:], axis=0)
        plt.figure()
        plt.plot(warped.shape[0] - histogram, color=(1,0,0))
        plt.imshow(warped, cmap='gray')
        plt.show()

if 'test_slide' in options:
    test_path = r"C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images"
    test_imgs = os.listdir(test_path)
    for i in test_imgs:
        img_path = test_path + '/' + i
        img = mpimg.imread(img_path)
        warped, undist, masked = image_pipeline(img)
        sliding_window(warped)
