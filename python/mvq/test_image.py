"""
Test to load and display an image.

"""

import os
import sys
import numpy as np
import cv2
from mvq import images_path


def load_image(image_file):

    # Load a color image
    img = cv2.imread(image_file, cv2.IMREAD_COLOR)

    cv2.rectangle(img, (100, 300), (200, 400), (0, 255, 255), 2)

    cv2.imshow('Picture of a road', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    filename = os.path.join(images_path, 'road.jpg')
    load_image(filename)
