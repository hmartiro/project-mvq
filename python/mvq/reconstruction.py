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
    filtered_img = lane_marker_filter(img, tau=15)

    # Erode the smaller features away, make lane lines thinner
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filtered_img = cv2.erode(filtered_img, kernel=kernel)

    # filtered_img = cv2.Laplacian(img, cv2.CV_8U)
    # filtered_img = np.uint8(np.abs(filtered_img))
    # filtered_img = cv2.Canny(filtered_img, 300, 400, apertureSize=3, L2gradient=True)
    # filtered_img = cv2.GaussianBlur(filtered_img, (5, 5), 0)
    # filtered_img = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    # filtered_img = cv2.Laplacian(filtered_img, cv2.CV_64F)
    # filtered_img = np.uint8(np.abs(filtered_img))
    # filtered_img = cv2.morphologyEx(filtered_img, cv2.MORPH_OPEN, kernel)
    # ret3, filtered_img = cv2.threshold(filtered_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # filtered_img = cv2.dilate(filtered_img, kernel=kernel)
    # filtered_img = cv2.Laplacian(filtered_img, cv2.CV_64F, scale=300)
    # filtered_img = np.uint8(np.abs(filtered_img))

    # Calculate hough lines
    # lines = cv2.HoughLines(image=filtered_img, rho=10, theta=np.pi*10/360, threshold=140, srn=10, stn=10)
    lines = cv2.HoughLinesP(image=filtered_img, rho=1, theta=np.pi*1/180, threshold=10, minLineLength=170, maxLineGap=120)

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

    vanishing_point = ransac_vanishing_point_detection(lines,5,200)

    cv2.circle(img_copy, vanishing_point, 1, (255, 0, 255), 8)


    # Draw the hough lines
    # for line in lines:
    #
    #     # Calculate line in cartesian
    #     rho, theta = line[0]
    #     a, b = np.cos(theta), np.sin(theta)
    #     x0, y0 = a * rho, b * rho
    #
    #     # Calculate points on line
    #     p1 = x0 + 1000 * (-b), y0 + 1000 * a
    #     p2 = x0 - 1000 * (-b), y0 - 1000 * a
    #
    #     # Round to integer
    #     p1 = [int(p) for p in p1]
    #     p2 = [int(p) for p in p2]
    #
    #     cv2.line(img_copy, tuple(p1), tuple(p2), (255, 0, 0), 1)

    # Plot
    plt.figure()
    plt.subplot(211), plt.imshow(img_copy, 'gray')
    plt.title('Input image with Hough lines')
    plt.subplot(212), plt.imshow(filtered_img, 'gray')
    plt.title('Lane marker detection filter')

    # TODO: find and return the vanishing point
    return -1, -1


def ransac_vanishing_point_detection(lines, distance, iterations):
    """
    Calculate the vanishing point of the road markers.

    :param lines: the lines defined as a [x1, y1, x2, y2] (4xN array, where N is the number of lines)
    :param distance: the distance (in pixels) to determine if a measurement is consistent
    :param iterations: the number of RANSAC iterations to use
    :return: Coordinates of the road vanishing point
    """

    N = np.size(lines,0)

    # Creating the place to store the maximum number of consistant lines
    max_num_consistent_lines = 0

    # Looping through all of the iterations to find the most consistent value
    for i in range(0,N):
        # Randomly choosing the lines
        random_indices = np.random.choice(N, 2, replace=False)
        i1 = random_indices[0]
        i2 = random_indices[1]
        x1, y1, x2, y2 = lines[i1]
        x3, y3, x4, y4 = lines[i2]

        # Find the intersection point
        x_intersect = ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) )/( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
        y_intersect = ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) )/( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )

        this_num_consistent = 0;
        # Find the distance between the intersection and all of the other lines
        for i2 in range(0,N):
            tx1, ty1, tx2, ty2 = lines[i2]
            this_distance = np.abs( (ty2-ty1)*x_intersect - (tx2-tx1)*y_intersect + tx2*ty1 - ty2*tx1 )/np.sqrt( (ty2-ty1)**2 +(tx2-tx1)**2 )
            if( this_distance < distance ):
                this_num_consistent += 1

        # If it's greater, make this the new x, y intersect
        if (this_num_consistent > max_num_consistent_lines):
            best_fit_x = x_intersect
            best_fit_y = y_intersect
            max_num_consistent_lines = this_num_consistent

        best_fit_x = int(best_fit_x)
        best_fit_y = int(best_fit_y)


    return (best_fit_x, best_fit_y)






# This part is run when the script is executed, but not imported
if __name__ == '__main__':

    imgL = cv2.imread(os.path.join(stereo_path, 'Color_cam1', '0000000000.png'), cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(os.path.join(stereo_path, 'Color_cam2', '0000000000.png'), cv2.IMREAD_GRAYSCALE)

    find_vanishing_point(imgL)

    # plot_disparity_map(imgL, imgR)
    # plot_edges(imgL)

    plt.show()
