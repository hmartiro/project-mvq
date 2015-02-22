"""

"""

import os
import numpy as np
import cv2
from mvq import stereo_path
from matplotlib import pyplot as plt

imgL = cv2.imread(os.path.join(stereo_path, 'Color_cam1', '0000000000.png'), cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread(os.path.join(stereo_path, 'Color_cam2', '0000000000.png'), cv2.IMREAD_GRAYSCALE)

# Disparity map
stereo = cv2.StereoBM_create(numDisparities=48, blockSize=15)
disparity = stereo.compute(imgL, imgR)

# plt.figure()
# plt.subplot(211), plt.imshow(disparity, 'gray')
# plt.subplot(212), plt.imshow(imgL, 'gray')

# Edge detection
edges = cv2.Canny(imgL, 240, 400)

# plt.figure()
# plt.subplot(211), plt.imshow(edges, 'gray')
# plt.subplot(212), plt.imshow(imgL, 'gray')

# Lane marking detector
# Adapted from
# https://marcosnietoblog.wordpress.com/2011/12/27/lane-markings-detection-and-vanishing-point-detection-with-opencv/


# Calculate hough lines
lines = cv2.HoughLines(image=edges, rho=5, theta=.02, threshold=200)

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

    cv2.line(imgL, tuple(p1), tuple(p2), (255, 0, 0), 1)

plt.figure()
plt.subplot(211), plt.imshow(edges, 'gray')
plt.subplot(212), plt.imshow(imgL, 'gray')

plt.show()
