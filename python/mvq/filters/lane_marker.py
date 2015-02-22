"""

"""

import numpy as np


def lane_marker_filter(img, tau, threshold=130):
    """
    Lane marking detector - filter for detecting lane markings.

    Adapted from:
        https://marcosnietoblog.wordpress.com/2011/12/27/lane-markings-detection-and-vanishing-point-detection-with-opencv/

    :param img: Input image
    :param tau: Expected width of lane markings
    :return: Filtered image, np array of uint8 type
    """

    img_in = img.astype(float)

    img_left = img_in[:, 2*tau:]
    img_right = img_in[:, :-2*tau]

    img_out = np.zeros(img_in.shape)
    img_out[:, tau:-tau] = img_in[:, tau:-tau] * 2

    img_out[:, tau:-tau] -= img_left
    img_out[:, tau:-tau] -= img_right
    img_out[:, tau:-tau] -= np.abs(img_left - img_right)

    # Clamp image
    img_out = np.clip(img_out, 0, 255)

    img_thresholded = np.zeros(img_out.shape, np.uint8)
    img_thresholded[img_out >= threshold] = 1

    return img_thresholded
