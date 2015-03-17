"""

"""

import os
import time
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
from PIL import Image
# from mvq.external.oclcq import quantize_image, colour_quantization
from mvq.stereo import calculate_disparity_map
from mvq.stereo import plot_disparity_map
from scipy.fftpack import dct, idct

from mvq import kitti_path
from mvq import reconstruction


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


def create_quantization_tables():

    q_luminance = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
    ],np.float32)

    q_chrominance = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
    ],np.float32)

    q3_luminance = np.zeros((8,8,8,100),np.uint8)
    q3_chrominance = np.zeros((8,8,8,100),np.uint8)

    for q in range(100):
        Q = q+1
        S = 5000/Q if (Q < 50) else 200-2*Q

        # print(Q,S)

        this_q_luminance = ((S*q_luminance+50)/100)
        this_q_chrominance = ((S*q_chrominance+50)/100)

        this_q_luminance[this_q_luminance < 1] = 1
        this_q_chrominance[this_q_chrominance > 1] = 1

        this_q_luminance[this_q_luminance > 255] = 255
        this_q_chrominance[this_q_chrominance > 255] = 255

        this_q_luminance = np.uint8(this_q_luminance)
        this_q_chrominance = np.uint8(this_q_chrominance)

        this_q3_luminance = np.zeros((8,8,8),np.uint8)
        this_q3_chrominance = np.zeros((8,8,8),np.uint8)

        this_q3_luminance[:,:,0] = this_q_luminance.copy()
        this_q3_chrominance[:,:,0] = this_q_chrominance.copy()

        for x in range(8):
            for y in range(8):
                for z in range(1,8):
                    # this_q3_luminance[y,x,z] = np.max([this_q_luminance[y,x],this_q_luminance[y,z],this_q_luminance[z,x]])
                    # this_q3_chrominance[y,x,z] = np.max([this_q_chrominance[y,x],this_q_chrominance[y,z],this_q_chrominance[z,x]])
                    this_q3_luminance[y,x,z] = this_q_luminance[y,x]
                    this_q3_chrominance[y,x,z] = this_q_chrominance[y,x]

        q3_luminance[:,:,:,q] = this_q3_luminance
        q3_chrominance[:,:,:,q] = this_q3_chrominance

    return q3_luminance, q3_chrominance


luminance_tables, chrominance_tables = create_quantization_tables()


def process_cube(img, weight, quality):
    # TODO: check to make sure that size of img, and q_tables are consistent

    img = img.copy()

    # print('process_cube input: {}'.format(img))

    this_quality = np.round(np.max(weight)*quality)

    if this_quality < 0:
        this_quality = 0
    if this_quality > quality - 1:
        this_quality = quality - 1

    for i in range(img.shape[3]):
        img[:, :, :, i] = cv2.cvtColor(img[:, :, :, i], cv2.COLOR_BGR2LAB)
    img = np.float32(img)

    # print('process_cube pre DCT: {}'.format(img))

    # img_dct = dct(dct(dct(img, axis=0)/4, axis=1)/4, axis=3)/4
    img_dct = dct(dct(img, axis=0)/4, axis=1)/4

    Q_luma = luminance_tables[:, :, :, this_quality].astype(np.float32)
    Q_chroma = chrominance_tables[:, :, :, this_quality].astype(np.float32)

    # Q_luma[:, :, :] = .01
    # Q_chroma[:, :, :] = .01

    # print('Q_luma: {}'.format(Q_luma))
    # print('Q_chroma: {}'.format(Q_chroma))

    # print('dct, pre rounding: {}'.format(img_dct))
    img_dct[:, :, 0, :] /= Q_luma
    img_dct[:, :, 1, :] /= Q_chroma
    img_dct[:, :, 2, :] /= Q_chroma

    img_dct = np.round(img_dct)

    img_dct[:, :, 0, :] *= Q_luma
    img_dct[:, :, 1, :] *= Q_chroma
    img_dct[:, :, 2, :] *= Q_chroma
    # print('dct, post rounding: {}'.format(img_dct))

    # img_processed = idct(idct(idct(img_dct, axis=0)/4, axis=1)/4, axis=3)/4
    img_processed = idct(idct(img_dct, axis=0)/4, axis=1)/4

    # print('process_cube post DCT: {}'.format(img_processed))

    img_processed = np.clip(img_processed, 0, 255)
    img_processed = np.uint8(img_processed)

    for i in range(img.shape[3]):
        img_processed[:,:,:,i] = cv2.cvtColor(img_processed[:,:,:,i], cv2.COLOR_LAB2BGR)

    # print('process_cube output: {}'.format(img))

    # print('pre dct / post_dct: {}'.format(pre_dct / post_dct))
    return img_processed


def process_block(img, weight, quality):

    img = img.copy()

    # print('process_cube input: {}'.format(img))

    this_quality = np.round(np.max(weight)*quality)

    if this_quality < 0:
        this_quality = 0
    if this_quality > quality - 1:
        this_quality = quality - 1

    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img = np.float32(img)

    img_dct = dct(dct(img, axis=0)/4, axis=1)/4

    Q_luma = luminance_tables[:, :, :, this_quality][:, :, 0].astype(np.float32)
    Q_chroma = chrominance_tables[:, :, :, this_quality][:, :, 0].astype(np.float32)

    img_dct[:, :, 0] /= Q_luma
    img_dct[:, :, 1] /= Q_chroma
    img_dct[:, :, 2] /= Q_chroma

    img_dct = np.round(img_dct)

    img_dct[:, :, 0] *= Q_luma
    img_dct[:, :, 1] *= Q_chroma
    img_dct[:, :, 2] *= Q_chroma

    img_processed = idct(idct(img_dct, axis=0)/4, axis=1)/4

    img_processed = np.clip(img_processed, 0, 255)
    img_processed = np.uint8(img_processed)

    img_processed = cv2.cvtColor(img_processed, cv2.COLOR_LAB2BGR)

    return img_processed


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

    random_values = np.random.rand(weights.shape[0], weights.shape[1])
    random_indices = random_values > weights

    img_random = img.copy()
    img_random[random_indices, :] = 0

    # Add bright colors
    augmented = np.zeros([img_random.shape[0], img_random.shape[1] + 50, img_random.shape[2]], dtype=np.uint8)
    augmented[:, :, 2] = 255
    augmented[:, :-50, :] = img_random

    augmented2 = np.zeros([augmented.shape[0], augmented.shape[1] + 50, augmented.shape[2]], dtype=np.uint8)
    augmented2[:, :, 0] = 250
    augmented2[:, :, 1] = 250
    augmented2[:, :, 2] = 180
    augmented2[:, :-50, :] = augmented

    i = Image.fromarray(augmented2)

    i = i.quantize(colors=num_colors, method=0)

    return i


def quantize_from_palette(img, palette):

    scale = 5
    out_im = cv2.resize(img, (img.shape[1]*scale, img.shape[0]*scale), interpolation=cv2.INTER_NEAREST)

    out_im = Image.fromarray(out_im)
    out_im = out_im.quantize(method=0, palette=palette).convert("RGB")
    out_im = np.asarray(out_im)

    out_im = cv2.medianBlur(out_im, 3)

    out_im = cv2.resize(out_im, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    return out_im


def combine_weights(W_list):

    W = np.zeros([W_list[0].shape[0], W_list[0].shape[1], len(W_list)])

    for i in range(len(W_list)):
        W[:, :, i] = W_list[i]

    W = np.max(W, axis=2)

    return W


def calculate_global_weight_matrix(img_left, img_right, show=False):

    W = np.zeros((img_left.shape[1], img_left.shape[0]))

    W_triangle = reconstruction.find_holy_triangle(img_left, -100, 850) + 0.02
    W_triangle = cv2.GaussianBlur(W_triangle, (101, 101), 0)

    W_disparity = calculate_disparity_map(img_left, img_right)
    W_disparity = cv2.GaussianBlur(W_disparity, (21, 21), 0)
    W_disparity_smoothed = cv2.GaussianBlur(W_disparity, (101, 101), 0)
    W_disparity = combine_weights([W_disparity, W_disparity_smoothed])
    W_disparity = cv2.GaussianBlur(W_disparity, (21, 21), 0)
    W_disparity = W_disparity / np.max(W_disparity)

    W = combine_weights([W_triangle, W_disparity])

    # if show:
    #     plt.figure()
    #     plt.subplot(311), plt.imshow(W_triangle, vmin=0, vmax=1)
    #     plt.title('W_triangle')
    #     plt.subplot(312), plt.imshow(W_disparity, vmin=0, vmax=1)
    #     plt.title('W_disparity')
    #     plt.subplot(313), plt.imshow(W, vmin=0, vmax=1)
    #     plt.title('W')

    return W


def process_block_using_imwrite(img, weight, quality):

    chroma_quality = np.mean(weight) * quality
    print(chroma_quality)

    chroma_quality = int(chroma_quality)
    if chroma_quality < 1:
        chroma_quality = 1

    luma_quality = chroma_quality + 1

    jpg_out_path = 'tmp.jpg'
    cv2.imwrite(jpg_out_path, img, params=(
        cv2.IMWRITE_JPEG_OPTIMIZE, 1,
        cv2.IMWRITE_JPEG_LUMA_QUALITY, luma_quality,
        cv2.IMWRITE_JPEG_CHROMA_QUALITY, chroma_quality,
        # cv2.IMWRITE_JPEG_QUALITY, 100,
        # cv2.IMWRITE_JPEG_PROGRESSIVE, 1
        # cv2.IMWRITE_JPEG_RST_INTERVAL, 100
    ))

    img = cv2.imread(jpg_out_path, cv2.IMREAD_ANYCOLOR)
    return img


def process_image(img_left, img_right, block_size=8):

    W = calculate_global_weight_matrix(img_left, img_right)

    img = img_left.copy()

    for i in range(int(img.shape[0]/block_size) + 1):
        for j in range(int(img.shape[1]/block_size) + 1):

            y1 = i * block_size
            x1 = j * block_size

            y2 = (i+1) * block_size
            x2 = (j+1) * block_size

            print('Block {})'.format((i, j)))

            if y2 > img.shape[0]:
                # y2 = img.shape[0]
                continue
            if x2 > img.shape[1]:
                # x2 = img.shape[1]
                continue

            block = process_block(img[y1:y2, x1:x2, :], W[y1:y2, x1:x2], quality=10)

            img[y1:y2, x1:x2, :] = block

    # img = img[0:int(img.shape[0]/block_size)*block_size, 0:int(img.shape[1]/block_size)*block_size, :]

    # smoothed_img = cv2.GaussianBlur(img, (5, 5), 0)
    # img[:, :, 0] = W * img[:, :, 0] + (1 - W) * smoothed_img[:, :, 0]
    # img[:, :, 1] = W * img[:, :, 1] + (1 - W) * smoothed_img[:, :, 1]
    # img[:, :, 2] = W * img[:, :, 2] + (1 - W) * smoothed_img[:, :, 2]
    #
    # img = cv2.bilateralFilter(img, 50, 75, 75)
    #
    # found_palette = quantize_find_palette(img, W, 25)
    # img = quantize_from_palette(img, found_palette)

    # img, gray_labels = quantize_kmeans(img, num_colors=20)
    # img = quantize_bit_shift(img, 3, 3, 3)

    return img


def main(show=False):

    from pathlib import Path

    # Get paths to stereo images
    left_path = str(Path(kitti_path) / 'left' / '0000000222.png')
    right_path = str(Path(kitti_path) / 'right' / '0000000222.png')

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
    img_out = process_image(img_left, img_right)

    # Save the data as an array
    # import pickle
    # import zlib
    # data_path = left_path[:-4] + '_data.out'
    # with open(data_path, 'wb') as f:
    #
    #     data = pickle.dumps(img_out)
    #     data = zlib.compress(data)
    #     f.write(data)

    # Save the output image
    out_path = left_path[:-4] + '_out.png'
    print('')
    print('Saving output image to "{}"'.format(out_path))
    cv2.imwrite(out_path, img_out, params=(
        cv2.IMWRITE_PNG_COMPRESSION, 9,
        cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_FILTERED,
        # cv2.IMWRITE_PNG_BILEVEL, True
    ))
    jpg_out_path = left_path[:-4] + '_out.jpg'
    cv2.imwrite(jpg_out_path, img_out, params=(
        cv2.IMWRITE_JPEG_OPTIMIZE, 1,
        cv2.IMWRITE_JPEG_LUMA_QUALITY, 11,
        cv2.IMWRITE_JPEG_CHROMA_QUALITY, 10,
        # cv2.IMWRITE_JPEG_QUALITY, 100,
        # cv2.IMWRITE_JPEG_PROGRESSIVE, 1
        # cv2.IMWRITE_JPEG_RST_INTERVAL, 100
    ))

    # Show the output image
    # if show:
    #     plt.figure()
    #     plt.subplot(211), plt.imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
    #     plt.title('Original')
    #     plt.subplot(212), plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    #     plt.title('Compressed')
    #     plt.show()

    # Compute the compression
    in_size = os.stat(in_path).st_size
    out_size = os.stat(out_path).st_size
    jpg_out_size = os.stat(jpg_out_path).st_size
    # data_file_size = os.stat(data_path).st_size
    compression_ratio = out_size / in_size

    print('')
    print('Input file size: {} bytes'.format(in_size))
    print('PNG Output file size: {} bytes'.format(out_size))
    print('JPG Output file size: {} bytes'.format(jpg_out_size))
    # print('Data file size: {} bytes'.format(data_file_size))
    print('Compression ratio: {}'.format(compression_ratio))


if __name__ == '__main__':

    import sys

    # Display the output only if the --show flag is used
    show_output = '--show' in sys.argv
    main(show=show_output)
