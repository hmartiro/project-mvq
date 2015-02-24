"""
Attempts at 3D scene reconstruction from a vehicle stereo camera pair.

"""

import os
import numpy as np
import cv2
from mvq import stereo_path
from matplotlib import pyplot as plt


def calculate_disparity_map(img_left, img_right):
    """
    Calculate and plot a disparity map of the stereo pair.
    """

    # Calculate the disparity map
    # For information on the algorithm, see
    # http://docs.opencv.org/ref/master/d2/d85/classcv_1_1StereoSGBM.html
    # stereo = cv2.StereoBM_create(numDisparities=48, blockSize=15)
    window_size = 3
    min_disp = 1
    max_disp = 16*4
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=max_disp,
        blockSize=window_size,
        uniquenessRatio=25,
        speckleWindowSize=50,
        speckleRange=32,
        disp12MaxDiff=1,
        P1=1*window_size**2,
        P2=32*3*window_size**2,
        mode=cv2.STEREO_SGBM_MODE_SGBM
    )

    disparity_map = stereo.compute(img_left, img_right).astype(np.float32) / 16

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    disparity_map = cv2.morphologyEx(disparity_map, cv2.MORPH_CLOSE, kernel)

    return disparity_map


def plot_disparity_map(img_left, img_right, disparity_map):

    # Plot
    plt.figure()
    plt.subplot(211), plt.imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
    plt.title('Input image (left) vs stereo disparity map')
    plt.subplot(212), plt.imshow(disparity_map, 'gray')

# This part is run when the script is executed, but not imported
if __name__ == '__main__':

    imgL = cv2.imread(os.path.join(stereo_path, 'Color_cam1', '0000000000.png'), cv2.IMREAD_COLOR)
    imgR = cv2.imread(os.path.join(stereo_path, 'Color_cam2', '0000000000.png'), cv2.IMREAD_COLOR)

    disparity = calculate_disparity_map(imgL, imgR)
    plot_disparity_map(imgL, imgR, disparity)

    plt.show()
