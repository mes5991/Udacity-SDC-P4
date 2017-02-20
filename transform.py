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
        img_path = folder_path + i
        img = mpimg.imread(img_path)
        undist = undistort(img, mtx, dist)
        warped, dst, src = transform(undist)

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

def transform(img):
    offset = 0
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[(img_size[0] / 2) - 65, img_size[1] / 2 + 100],
                      [((img_size[0] / 6) - 50), img_size[1]],
                      [(img_size[0] * 5 / 6) + 60, img_size[1]],
                      [(img_size[0] / 2 + 65), img_size[1] / 2 + 100]])
    dst = np.float32([[(img_size[0] / 4), 0],
                      [(img_size[0] / 4), img_size[1]],
                      [(img_size[0] * 3 / 4), img_size[1]],
                      [(img_size[0] * 3 / 4), 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size)
    return warped, dst, src

def display_image(img):
    plt.imshow(img)
    plt.show()

def display_image_with_lines(img):
    offset = 20
    # img = cv2.line(img, (215-offset,700), (615-offset,435), color=(255,0,0), thickness=3)
    # img = cv2.line(img, (1050+offset,680), (661+offset,435), color=(255,0,0), thickness=3)

    img = cv2.line(img, (585-offset,460), (203-offset,720), color=(255,0,0), thickness=3)
    img = cv2.line(img, (1127+offset,720), (695+offset,460), color=(255,0,0), thickness=3)
    img = cv2.line(img, (585-offset,460), (695+offset,460), color=(255,0,0), thickness=3)
    img = cv2.line(img, (203-offset,720), (1127+offset,720), color=(255,0,0), thickness=3)
    plt.imshow(img)
    plt.show()

def draw_lines(img, points, color):
    img = cv2.line(img, (points[0][0], points[0][1]), (points[1][0], points[1][1]), color=color, thickness=3)
    img = cv2.line(img, (points[1][0], points[1][1]), (points[2][0], points[2][1]), color=color, thickness=3)
    img = cv2.line(img, (points[2][0], points[2][1]), (points[3][0], points[3][1]), color=color, thickness=3)
    img = cv2.line(img, (points[3][0], points[3][1]), (points[0][0], points[0][1]), color=color, thickness=3)
    return img
