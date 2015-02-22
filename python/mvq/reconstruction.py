"""
Attempts at 3D scene reconstruction from a vehicle stereo camera pair.

"""

import os
import numpy as np
import cv2
from mvq import stereo_path
from matplotlib import pyplot as plt

from mvq.filters.lane_marker import lane_marker_filter


def plot_edges(img):
    """
    Calculate and plot edges of the image.
    """

    edges = cv2.Canny(img, 240, 400)

    # Plot
    plt.figure()
    plt.subplot(211), plt.imshow(img, 'gray')
    plt.title('Input image')
    plt.subplot(212), plt.imshow(edges, 'gray')
    plt.title('Edges')


def plot_disparity_map(img_left, img_right):
    """
    Calculate and plot a disparity map of the stereo pair.
    """

    # Calculate the disparity map
    stereo = cv2.StereoBM_create(numDisparities=48, blockSize=15)
    disparity = stereo.compute(img_left, img_right)

    # Plot
    plt.figure()
    plt.subplot(211), plt.imshow(img_left, 'gray')
    plt.title('Input image (left)')
    plt.subplot(212), plt.imshow(disparity, 'gray')
    plt.title('Disparity map')


def find_vanishing_point(img):
    """
    Calculate the vanishing point of the road markers.

    :param img: Image of a road from a driving car
    :return: Coordinates of the road vanishing point
    """

    # Filter to find the lane markers
    filtered_img = lane_marker_filter(img, tau=15, threshold=130)

    # Calculate hough lines
    lines = cv2.HoughLines(image=filtered_img, rho=20, theta=.01, threshold=700)

    # Create a copy of the input to draw lines on
    img_copy = np.array(img)

    # Draw the hough lines
    for line in lines:

        # Calculate line in cartesian
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho

        # Calculate points on line
        p1 = x0 + 1000 * (-b), y0 + 1000 * a
        p2 = x0 - 1000 * (-b), y0 - 1000 * a

        # Round to integer
        p1 = [int(p) for p in p1]
        p2 = [int(p) for p in p2]

        cv2.line(img_copy, tuple(p1), tuple(p2), (255, 0, 0), 1)

    # Plot
    plt.figure()
    plt.subplot(211), plt.imshow(img_copy, 'gray')
    plt.title('Input image with Hough lines')
    plt.subplot(212), plt.imshow(filtered_img, 'gray')
    plt.title('Lane marker detection filter')

    # TODO: find and return the vanishing point
    return -1, -1

# This part is run when the script is executed, but not imported
if __name__ == '__main__':

    imgL = cv2.imread(os.path.join(stereo_path, 'Color_cam1', '0000000000.png'), cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(os.path.join(stereo_path, 'Color_cam2', '0000000000.png'), cv2.IMREAD_GRAYSCALE)

    find_vanishing_point(imgL)

    plot_disparity_map(imgL, imgR)
    plot_edges(imgL)

    plt.show()
