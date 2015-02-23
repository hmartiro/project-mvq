"""

"""

import numpy as np
import cv2


def lane_marker_filter(img, tau):
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

    img_out[:, tau:-tau] -= img_left + img_right
    img_out[:, tau:-tau] -= np.abs(img_left - img_right)

    # Cut off the top third
    cutoff_min_y = int(img_out.shape[0] * 1/3)
    img_out[0:cutoff_min_y, :] = 0

    # Clamp image
    img_out = np.clip(img_out, 0, 255)

    # Otsu thresholding
    img_out = np.uint8(img_out)
    ret3, img_out = cv2.threshold(img_out, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # img_thresholded = np.zeros(img_out.shape, np.uint8)
    # img_thresholded[img_out >= threshold] = 1
    # img_out = img_thresholded

    return img_out
