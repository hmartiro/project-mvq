"""

"""

import os
import time
import numpy as np
import math
import cv2
from mvq import stereo_path
from mvq import reconstruction
from PIL import Image


def quantize_bit_shift(img, shift_r, shift_g, shift_b):

    shift_b = 8 - shift_b
    shift_g = 8 - shift_g
    shift_r = 8 - shift_r

    img[:, :, 0] >>= shift_b
    img[:, :, 1] >>= shift_g
    img[:, :, 2] >>= shift_r

    img[:, :, 0] <<= shift_b
    img[:, :, 1] <<= shift_g
    img[:, :, 2] <<= shift_r

    return img


def compression_test(img):

    # img = quantize_bit_shift(img, 2, 3, 3)

    img[:, 0:250, :] = quantize_bit_shift(img[:, 0:250, :], 2, 2, 3)

    img[:, 250:500, :] = quantize_bit_shift(img[:, 250:500, :], 2, 3, 2)
    # img[:, 250:500, :] += 1

    img[:, 500:750, :] = quantize_bit_shift(img[:, 500:750, :], 3, 2, 2)
    # img[:, 500:750, :] += 2

    img[:, 750:1000, :] = quantize_bit_shift(img[:, 750:1000, :], 2, 2, 3)
    img[:, 750:1000, :] += 1

    img[0:, 1000:, :] = quantize_bit_shift(img[0:, 1000:, :], 2, 2, 3)
    img[0:, 1000:, :] += 2
    #img[0:50, 1000:, :] = np.random.randint(0, 255, [50, img.shape[1]-1000, 3])
    img[0:50, 1000:, :] = [100, 0, 100]


def quantize_kmeans(img, num_colors):

    img_lab = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))

    img_lab = img_lab.reshape((img.shape[0] * img.shape[1], 3))

    num_samples = 10000
    random_indices = np.random.choice(img_lab.shape[0], num_samples, replace=False)
    data = img_lab[random_indices, :]

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS

    compactness, labels, centers = cv2.kmeans(
        data=data,
        K=num_colors,
        bestLabels=None,
        attempts=10,
        criteria=criteria,
        flags=flags
    )

    gray_labels = np.zeros([img.shape[0] * img.shape[1], 1], dtype=np.uint8)

    for i, p in enumerate(img_lab):

        diff = centers - p
        dist = np.sqrt(np.sum(diff**2, axis=1))

        min_center = np.argmin(dist)

        p[:] = centers[min_center]
        gray_labels[i] = min_center

    gray_labels = gray_labels.reshape(img.shape[0], img.shape[1])
    img_lab = img_lab.reshape(img.shape[0], img.shape[1], 3)

    img_lab = np.clip(img_lab, 0, 255)
    img_lab = np.uint8(img_lab)

    img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

    return img, gray_labels


def quantize_find_palette(img, weights, num_colors):

    random_values = np.random.rand(weights.shape[0],weights.shape[1])
    random_indices = random_values > weights

    img_random = img.copy()
    img_random[random_indices,:] = 0

    i = Image.fromarray(img_random.copy())
    i = i.quantize(colors=num_colors,method=0)

    return i

def quantize_from_palette(img, palette_im):
    out_im = Image.fromarray(img.copy())
    out_im = out_im.quantize(palette=palette_im).convert("RGB")

    return np.asarray(out_im)



def process(img_left, img_right):

    img = img_left[:, :]

    # img, gray_labels = quantize_kmeans(img, num_colors=256)
    # img = quantize_bit_shift(img, 3, 3, 3)
    weightings = reconstruction.find_holy_triangle(img, -100, 850) + 0.02
    found_palette = quantize_find_palette(img, weightings, 20)
    img = quantize_from_palette(img, found_palette)

    return img


def main(show=False):

    # Get paths to stereo images
    left_path = os.path.join(stereo_path, 'Color_cam1', '0000000000.png')
    right_path = os.path.join(stereo_path, 'Color_cam2', '0000000000.png')

    # Read in stereo images
    print('Reading left image from "{}"'.format(left_path))
    print('Reading right image from "{}"'.format(right_path))
    img_left = cv2.imread(left_path, cv2.IMREAD_COLOR)
    img_right = cv2.imread(right_path, cv2.IMREAD_COLOR)

    # Save original to compare with same save settings. This is so our
    # compression ratio at the end compares apples to apples.
    in_path = left_path[:-4] + '_in.png'
    print('')
    print('Saving original (left) to "{}"'.format(in_path))
    cv2.imwrite(in_path, img_left)

    # Process (compress)
    print('')
    print('Running smart compression...')
    img_out = process(img_left, img_right)

    # Save the data as an array
    import pickle
    import zlib
    data_path = left_path[:-4] + '_data.out'
    with open(data_path, 'wb') as f:

        data = pickle.dumps(img_out)
        data = zlib.compress(data)
        f.write(data)

    # Save the output image
    out_path = left_path[:-4] + '_out.png'
    print('')
    print('Saving output image to "{}"'.format(out_path))
    cv2.imwrite(out_path, img_out, params=(
        # cv2.IMWRITE_PNG_COMPRESSION, 9,
        # cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_FILTERED,
        # cv2.IMWRITE_PNG_BILEVEL, True
    ))

    # Show the output image
    if show:
        cv2.imshow('image', img_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Compute the compression
    in_size = os.stat(in_path).st_size
    out_size = os.stat(out_path).st_size
    data_file_size = os.stat(data_path).st_size
    compression_ratio = out_size / in_size

    print('')
    print('Input file size: {} bytes'.format(in_size))
    print('Output file size: {} bytes'.format(out_size))
    print('Data file size: {} bytes'.format(data_file_size))
    print('Compression ratio: {}'.format(compression_ratio))


if __name__ == '__main__':

    import sys

    # Display the output only if the --show flag is used
    show_output = '--show' in sys.argv
    main(show=show_output)
