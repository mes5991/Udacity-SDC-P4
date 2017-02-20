import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def test_transform(folder_path, mtx, dist):
    imgs = os.listdir(folder_path)
    for i in imgs:
        #Load image
        img_path = folder_path + i
        img = mpimg.imread(img_path)

        #Undistort image based on camera calibration
        undist = undistort(img, mtx, dist)

        #Transform perspective to birds-eye-view
        warped, dst, src = transform2(undist)

        #Draw transformation lines and plot
        undist = draw_lines(undist, src, (255,0,0))
        undist = draw_lines(undist, dst, (0,255,0))
        warped = draw_lines(warped, dst, (255,0,0))
        plt.subplot(1,3,1)
        plt.imshow(img)
        plt.title("Original Image")
        plt.subplot(1,3,2)
        plt.imshow(undist)
        plt.title("Unwarped image with transform lines")
        plt.subplot(1,3,3)
        plt.imshow(warped)
        plt.title("Transformed image with transform lines")
        plt.show()

def transform1(img):
    offset = 0 #for quick modification
    img_size = (img.shape[1], img.shape[0])

    #points in the real space
    src = np.float32([[(img_size[0] / 2) - 65, img_size[1] / 2 + 100],
                      [((img_size[0] / 6) - 50), img_size[1]],
                      [(img_size[0] * 5 / 6) + 60, img_size[1]],
                      [(img_size[0] / 2 + 65), img_size[1] / 2 + 100]])

    #points on the image plane
    dst = np.float32([[(img_size[0] / 4), 0],
                      [(img_size[0] / 4), img_size[1]],
                      [(img_size[0] * 3 / 4), img_size[1]],
                      [(img_size[0] * 3 / 4), 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size)
    return warped, dst, src

def transform2(img):
    offset = 0 #for quick modification
    img_size = (img.shape[1], img.shape[0])

    #points in the real space
    src = np.float32([[580,460],
                      [700,460],
                      [1040,680],
                      [260,680]])

    #points on the image plane
    dst = np.float32([[260,0],
                      [1040,0],
                      [1040,720],
                      [260,720]])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size)
    return warped, dst, src

def display_image(img):
    plt.imshow(img)
    plt.show()

def draw_lines(img, points, color):
    img = cv2.line(img, (points[0][0], points[0][1]), (points[1][0], points[1][1]), color=color, thickness=3)
    img = cv2.line(img, (points[1][0], points[1][1]), (points[2][0], points[2][1]), color=color, thickness=3)
    img = cv2.line(img, (points[2][0], points[2][1]), (points[3][0], points[3][1]), color=color, thickness=3)
    img = cv2.line(img, (points[3][0], points[3][1]), (points[0][0], points[0][1]), color=color, thickness=3)
    return img
