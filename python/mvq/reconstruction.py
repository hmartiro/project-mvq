"""
Attempts at 3D scene reconstruction from a vehicle stereo camera pair.

"""

import os
import time
import numpy as np
import cv2
from mvq import stereo_path
from matplotlib import pyplot as plt

from mvq.filters.lane_marker import lane_marker_filter
from mvq.detectors.ransac_vanishing_point import ransac_vanishing_point_detection


def find_vanishing_point(img):
    """
    Calculate the vanishing point of the road markers.

    :param img: Image of a road from a driving car
    :return: Coordinates of the road vanishing point
    """

    # Filter to find the lane markers
    filtered_img = lane_marker_filter(img, tau=15)

    # Erode the smaller features away, make lane lines thinner
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filtered_img = cv2.erode(filtered_img, kernel=kernel)

    # Calculate hough lines
    lines = cv2.HoughLinesP(
        image=filtered_img,
        rho=1,
        theta=np.pi*1/180,
        threshold=10,
        minLineLength=170,
        maxLineGap=120
    )

    if lines is None:
        lines = []

    # Create a copy of the input to draw lines on
    img_copy = np.array(img)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)

    lines = [line[0] for line in lines]

    for line in lines:
        x1, y1, x2, y2 = line
        p1 = (x1, y1)
        p2 = (x2, y2)
        cv2.line(img_copy, p1, p2, (255, 255, 0), 2)

    vanishing_point = ransac_vanishing_point_detection(lines, 5, 20)

    # Plot the vanishing point
    cv2.circle(img_copy, vanishing_point, 1, (255, 0, 0), 8)

    # Plot
    plt.figure()
    plt.subplot(211), plt.imshow(img_copy, 'gray')
    plt.title('Input image with Hough lines and vanishing point')
    plt.subplot(212), plt.imshow(filtered_img, 'gray')
    plt.title('Lane marker detection filter')


# This part is run when the script is executed, but not imported
if __name__ == '__main__':

    imgL = cv2.imread(os.path.join(stereo_path, 'Color_cam1', '0000000000.png'), cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(os.path.join(stereo_path, 'Color_cam2', '0000000000.png'), cv2.IMREAD_GRAYSCALE)

    find_vanishing_point(imgL)
    plt.show()
