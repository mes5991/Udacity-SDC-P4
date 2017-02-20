import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from calibration import *
from transform import *

#Calibrate camera
ret, mtx, dist, rvecs, tvecs = calibrate_camera(plot=False)
print("Camera Calibrated!")

#Test calibration
# test_calibration(mtx, dist)
# test_calibration_single_image(mtx, dist, r"C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images\straight_lines1.jpg")
# test_calibration_single_image(mtx, dist, r"C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images\straight_lines2.jpg")
# print("Calibration Tested!")

img_path = r"C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images\straight_lines1.jpg"
img_path = r"C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images\straight_lines2.jpg"
img_path = r"C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images\test1.jpg"
img = mpimg.imread(img_path)
undist = undistort(img, mtx, dist)
warped, dst, src = transform(undist)
print (dst)
warped = draw_lines(warped, dst, (255,0,0))
undist = draw_lines(undist, src, (255,0,0))
undist = draw_lines(undist, dst, (0,255,0))
# display_image_with_lines(undist)
plt.subplot(1,3,1)
plt.imshow(img)
plt.subplot(1,3,2)
plt.imshow(undist)
plt.subplot(1,3,3)
plt.imshow(warped)
plt.show()

display_image(undist)
