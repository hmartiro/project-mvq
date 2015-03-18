"""
Vanishing point and line marker estimation from a single camera image.

"""

import os
import time
import numpy as np
import cv2
from mvq import stereo_path
from matplotlib import pyplot as plt

from mvq.filters.lane_marker import lane_marker_filter
from mvq.detectors.ransac_vanishing_point import ransac_vanishing_point_detection


def find_vanishing_point(img, show=False):
    """
    Calculate the vanishing point of the road markers.

    :param img: Image of a road from a driving car
    :return: Coordinates of the road vanishing point
    """

    # Filter to find the lane markers
    filtered_img = lane_marker_filter(img, tau=10)

    # Erode the smaller features away, make lane lines thinner
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filtered_img = cv2.erode(filtered_img, kernel=kernel)

    # Calculate hough lines
    lines = cv2.HoughLinesP(
        image=filtered_img,
        rho=1,
        theta=np.pi*1/180,
        threshold=10,
        minLineLength=120,
        maxLineGap=80
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

    if vanishing_point is None:
        vanishing_point = 650, 175

    vanishing_point = vanishing_point[0], (vanishing_point[1] - 50)
    print(vanishing_point)

    # Plot the vanishing point
    cv2.circle(img_copy, vanishing_point, 1, (255, 0, 0), 20)

    if show:
        # Plot
        plt.figure()
        plt.subplot(211), plt.imshow(img_copy, 'gray')
        plt.title('Input image with Hough lines and vanishing point')
        plt.subplot(212), plt.imshow(filtered_img, 'gray')
        plt.title('Lane marker detection filter')

    return vanishing_point


def find_holy_triangle(img, left_x, right_x):
    vanishing_point = find_vanishing_point(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

    max_y = img.shape[0]

    out_im = np.zeros( (img.shape[0], img.shape[1]) )
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            left_line1 = (np.sign( (vanishing_point[0]-right_x)*(y-max_y) - (vanishing_point[1]-max_y)*(x-right_x) ) == -1) # left of line 1
            right_line2 = (np.sign( (vanishing_point[0]-left_x)*(y-max_y) - (vanishing_point[1]-max_y)*(x-left_x) ) == 1) # right of line 2
            if (left_line1 & right_line2):
                out_im[y,x] = 1.0

    return out_im

# This part is run when the script is executed, but not imported
if __name__ == '__main__':

    imgL = cv2.imread(os.path.join(stereo_path, 'Color_cam1', '0000000000.png'), cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(os.path.join(stereo_path, 'Color_cam2', '0000000000.png'), cv2.IMREAD_GRAYSCALE)

    find_vanishing_point(imgL)
    plt.show()
