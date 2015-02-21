"""
Playing around with OpenCV.

"""

import os
import sys
import numpy as np
import cv2


def load_image(image_file):

    # Load a color image
    img = cv2.imread(image_file, cv2.IMREAD_COLOR)
    cv2.imshow('Picture of a road', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    filename = '../../data/images/road.jpg'
    load_image(filename)
