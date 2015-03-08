"""
Module for creating and outputting videos.

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

from mvq import kitti_path
from mvq.stereo import calculate_disparity_map


def process_image(left_image, right_image, out_images):

    img = left_image.copy()

    img = cv2.bilateralFilter(img, 30, 75, 75)

    # img = cv2.medianBlur(img, 5)

    # return img

    # W = cv2.Canny(img, 50, 300)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    # W = cv2.morphologyEx(W, cv2.MORPH_DILATE, kernel)
    # W = cv2.GaussianBlur(W, (15, 15), 0)
    #
    # W = W / np.max(W)
    # return W * 255

    # W = calculate_disparity_map(left_image, right_image)
    # W = cv2.GaussianBlur(W, (21, 21), 0)
    # W = W / np.max(W)
    #
    # smoothed_img = cv2.GaussianBlur(img, (21, 21), 0)
    # img[:, :, 0] = W * img[:, :, 0] + (1 - W) * smoothed_img[:, :, 0]
    # img[:, :, 1] = W * img[:, :, 1] + (1 - W) * smoothed_img[:, :, 1]
    # img[:, :, 2] = W * img[:, :, 2] + (1 - W) * smoothed_img[:, :, 2]



    # img[:, :, 0] = W * img[:, :, 0] + (1 - W) * bilateral_img[:, :, 0]
    # img[:, :, 1] = W * img[:, :, 1] + (1 - W) * bilateral_img[:, :, 1]
    # img[:, :, 2] = W * img[:, :, 2] + (1 - W) * bilateral_img[:, :, 2]

    # plt.figure()
    # plt.subplot(211), plt.imshow(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB))
    # plt.title('Original')
    # plt.subplot(212), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.title('Filtered')
    # plt.show()

    # input('Press to continue...')

    if len(out_images) == 0:
        return img

    # last_img = out_images[-1].astype(np.float32)
    #
    # img[:, :, 0] = W * img[:, :, 0] + (1 - W) * last_img[:, :, 0]
    # img[:, :, 1] = W * img[:, :, 1] + (1 - W) * last_img[:, :, 1]
    # img[:, :, 2] = W * img[:, :, 2] + (1 - W) * last_img[:, :, 2]


    # avg = (1-w) * img.astype(np.float32) + w * out_images[-1].astype(np.float32)

    # avg = np.clip(avg, 0, 255)
    # avg = np.uint8(avg)

    # avg = smooth(avg, size=10)

    return img


def process(left_image_paths, right_image_paths, out_filename):

    N = len(left_image_paths)
    assert(len(left_image_paths) == len(right_image_paths))

    img0 = cv2.imread(left_image_paths[0], cv2.IMREAD_COLOR)
    frame_size = (img0.shape[1], img0.shape[0])

    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(
        filename=out_filename,
        fourcc=fourcc,
        fps=fps,
        frameSize=frame_size
    )

    out_images = []

    for i in range(N):

        print('i = {}, processing image {}'.format(i, left_image_paths[i][-14:]))

        left_image = cv2.imread(left_image_paths[i], cv2.IMREAD_COLOR)
        right_image = cv2.imread(right_image_paths[i], cv2.IMREAD_COLOR)

        out_image = process_image(left_image, right_image, out_images)

        out_image = np.clip(out_image, 0, 255)
        out_image = np.uint8(out_image)

        out.write(out_image)

        cv2.imshow('video', out_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out_images.append(out_image)

    out.release()
    cv2.destroyAllWindows()


def main():

    from pathlib import Path

    p = Path(kitti_path)

    left_image_paths = list(str(x) for x in (p / 'left').glob('*.png'))
    left_image_paths.sort()
    print('Compressing {} images'.format(len(left_image_paths)))

    right_image_paths = list(str(x) for x in (p / 'right').glob('*.png'))
    right_image_paths.sort()

    process(left_image_paths[65:], right_image_paths[65:], 'output.avi')


if __name__ == '__main__':
    main()
