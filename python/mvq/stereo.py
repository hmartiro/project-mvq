"""
Disparity map calculation from the vehicle stereo camera pair.

"""

import os
import numpy as np
import cv2
from mvq import kitti_path
from matplotlib import pyplot as plt

window_size = 2
min_disp = 5
max_disp = 16*4
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=max_disp,
    blockSize=window_size,
    uniquenessRatio=25,
    # speckleWindowSize=50,
    # speckleRange=32,
    # disp12MaxDiff=1,
    P1=1*window_size**2,
    P2=32*7*window_size**2,
    mode=cv2.STEREO_SGBM_MODE_HH
)


def calculate_disparity_map(img_left, img_right):
    """
    Calculate and plot a disparity map of the stereo pair.
    """

    # Calculate the disparity map
    # For information on the algorithm, see
    # http://docs.opencv.org/ref/master/d2/d85/classcv_1_1StereoSGBM.html
    # stereo = cv2.StereoBM_create(numDisparities=48, blockSize=15)

    disparity_map = stereo.compute(img_left, img_right).astype(np.float32)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    disparity_map = cv2.morphologyEx(disparity_map, cv2.MORPH_CLOSE, kernel)

    return disparity_map


def plot_disparity_map(img_left, img_right, disparity_map):

    # Plot
    plt.figure()
    plt.subplot(311), plt.imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
    plt.title('Left camera view')
    plt.subplot(312), plt.imshow(cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB))
    plt.title('Right camera view')
    plt.subplot(313), plt.imshow(disparity_map, 'gray')
    plt.title('Calculated disparity map')

# This part is run when the script is executed, but not imported
if __name__ == '__main__':

    imgL = cv2.imread(os.path.join(kitti_path, 'left', '0000000104.png'), cv2.IMREAD_COLOR)
    imgR = cv2.imread(os.path.join(kitti_path, 'right', '0000000104.png'), cv2.IMREAD_COLOR)

    disparity = calculate_disparity_map(imgL, imgR)
    plot_disparity_map(imgL, imgR, disparity)

    plt.show()
