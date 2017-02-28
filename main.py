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

def image_pipeline(img):
    #Undistort the image
    undist = undistort(img, mtx, dist)
    undist = cv2.GaussianBlur(undist, (5,5), 0)
    undist = normalize(undist)

    #Warp perspective
    warped, dst, src, Minv = transform1(undist)

    #Threshold to generate binary image where lane lines are clear
    threshold_img = thresh_pipeline(warped)[0]
    # result = threshold_img
    kernel = np.ones((5,5),np.uint8)
    result = cv2.morphologyEx(threshold_img, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((10,10),np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    return result, threshold_img, warped, undist, Minv

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
        img = mpimg.imread(img_path)
        undist = undistort(img, mtx, dist)
        undist = cv2.GaussianBlur(undist, (5,5), 0)
        undist = normalize(undist)
        warped, dst, src = transform1(undist)

        combined_binary, gradx, sbinary, gray_binary, morphed, masked, mag_binary = thresh_pipeline(warped)

        plt.subplot(3,3,1)
        plt.imshow(warped)
        plt.title("warped")
        plt.subplot(3,3,2)
        plt.imshow(gradx, cmap='gray')
        plt.title("grax")
        plt.subplot(3,3,3)
        plt.imshow(sbinary, cmap='gray')
        plt.title("sbinary")
        plt.subplot(3,3,4)
        plt.imshow(gray_binary, cmap='gray')
        plt.title("gray_binary")
        plt.subplot(3,3,5)
        plt.imshow(masked, cmap='gray')
        plt.title("gray_binary_morphed")
        plt.subplot(3,3,6)
        plt.imshow(mag_binary, cmap='gray')
        plt.title("mag_binary")
        plt.subplot(3,3,7)
        plt.imshow(combined_binary, cmap='gray')
        plt.title("combined_binary")
        plt.show()

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
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    videoout = cv2.VideoWriter('output.avi',fourcc, 30.0,( 1280,720))
    # left_fit = None
    # right_fit = None
    left_line = Line()
    right_line = Line()
    i=0
    while(cap.isOpened()):
        i += 1
        ret, frame = cap.read()
        # print(i)
        # if i < 500:
        #     continue
        result, threshold_img, warped, undist, Minv = image_pipeline(frame)
        # result[result > 0] = 255

        # plt.ion()
        green_space = sliding_window(result, warped, left_line, right_line, plot = False)
        green_space_transform = cv2.warpPerspective(green_space, Minv, (green_space.shape[1], green_space.shape[0]))
        output = cv2.addWeighted(frame, 1, green_space_transform, 0.3, 0)

        videoout.write(output)
        # cv2.imshow('Video', frame)
    #     cv2.imshow('output', output)
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
        result, threshold_img, warped, undist = image_pipeline(img)
        sliding_window(result)
