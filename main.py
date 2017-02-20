import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from calibration import *
from transform import *

options = sys.argv

#Calibrate camera
ret, mtx, dist, rvecs, tvecs = calibrate_camera(plot=False)
print("Camera Calibrated!")

#Test calibration
if 'test_calibration' in options:
    test_calibration(mtx, dist)
    # test_calibration_single_image(mtx, dist, r"C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images\straight_lines1.jpg")
    # test_calibration_single_image(mtx, dist, r"C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images\straight_lines2.jpg")
    print("Calibration Tested!")

if 'test_transform' in options:
    test_transform(r'C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 4\CarND-Advanced-Lane-Lines\test_images\\', mtx, dist)
    print("Transforms tested!")
