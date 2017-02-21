import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from calibration import *
from transform import *
from imageProcessing import *

options = sys.argv

#Calibrate camera
ret, mtx, dist, rvecs, tvecs = calibrate_camera(plot=False)
print("Camera Calibrated!")

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
        test_sobel_thresh(img)

if 'test_color_thresh' in options:
    test_path = r"C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images"
    test_imgs = os.listdir(test_path)
    for i in test_imgs:
        img_path = test_path + '/' + i
        test_color_thresholding(img_path, (160,255))

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

def image_pipeline(img):
    #Undistort the image
    undist = undistort(img, mtx, dist)
    undist = cv2.GaussianBlur(undist, (5,5), 0)

    #Threshold to generate binary image where lane lines are clear
    threshold_img = thresh_pipeline(undist)

    #Warp perspective
    warped, dst, src = transform1(threshold_img)

    return warped, undist, threshold_img

if 'test_pipeline' in options:
    test_path = r"C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images"
    test_imgs = os.listdir(test_path)
    for i in test_imgs:
        img_path = test_path + '/' + i
        img = mpimg.imread(img_path)
        warped, undist, threshold_img = image_pipeline(img)

        plt.figure(figsize=(20,10))
        plt.subplot(2,2,1)
        plt.imshow(img)
        plt.subplot(2,2,2)
        plt.imshow(undist)
        plt.subplot(2,2,3)
        plt.imshow(threshold_img, cmap='gray')
        plt.subplot(2,2,4)
        plt.imshow(warped, cmap='gray')
        plt.show()
