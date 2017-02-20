import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def calibrate_camera(plot=False):
    calibration_path = "C:/Users/mes59/Documents/Udacity/SDC/Term 1/Project 4/CarND-Advanced-Lane-Lines/camera_cal/calibration"

    objpoints = [] #3D points in real world space
    imgpoints = [] #2D points in image plane

    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2) #Represent the coordinates of the real grid. This is the same for all images

    for i in range(1,21):
        # if (i == 1):
        #     n_corners = (9,5)
        # elif (i == 4):
        #     continue
        # elif (i==5):
        #     n_corners = (7,5)
        # else:
        #     n_corners = (9,6)

        n_corners = (9,6)

        #Read in image
        fname = calibration_path + str(i) + '.jpg'
        img = cv2.imread(fname)

        #Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, n_corners, None)

        #If corners are found, add object points and image points
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

            #Display corners on image
            if plot:
                cv2.drawChessboardCorners(img, n_corners, corners, ret)
                plt.imshow(img)
                plt.show()
        else:
            print("Failed to find corners on image: ", + i)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    return ret, mtx, dist, rvecs, tvecs

def test_calibration(mtx, dist):
    calibration_path = "C:/Users/mes59/Documents/Udacity/SDC/Term 1/Project 4/CarND-Advanced-Lane-Lines/camera_cal/calibration"
    for i in range(1,21):
        fname = calibration_path + str(i) + '.jpg'
        img = cv2.imread(fname)
        undist = cv2.undistort(img, mtx, dist, None, mtx)

        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.subplot(1,2,2)
        plt.imshow(undist)
        plt.show()

def test_calibration_single_image(mtx, dist, img_path):
    img = mpimg.imread(img_path)
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.imshow(undist)
    plt.show()
