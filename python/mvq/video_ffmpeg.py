"""

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mvq.compress import process_cube
from mvq.compress import create_quantization_tables
from mvq.compress import calculate_global_weight_matrix
from mvq.compress import process_image

from mvq import kitti_path

# Create quantization tables
q3_luminance, q3_chrominance = create_quantization_tables()


def process_image_block(left_images, right_images):

    img_shape = left_images[0].shape
    block_size = len(left_images)

    W_3d = np.zeros([img_shape[0], img_shape[1], block_size])

    for i in range(block_size):
        W_3d[:, :, i] = calculate_global_weight_matrix(left_images[i], right_images[i])
        # W_3d[:, :, i] = np.ones((left_images[i].shape[0], left_images[i].shape[1]), np.float32) * 0.8

    images_3d = np.zeros(img_shape + (block_size,), dtype=left_images[0].dtype)
    for i in range(block_size):
        images_3d[:, :, :, i] = left_images[i]
    # print('images_3d dtype: {}'.format(images_3d.dtype))

    out_3d = np.zeros(images_3d.shape, dtype=np.uint8)
    # print('out_3d dtype: {}'.format(out_3d.dtype))

    # Break up into 8x8 spacial blocks
    block_size = 8
    for i in range(int(out_3d.shape[0]/block_size) + 1):
        for j in range(int(out_3d.shape[1]/block_size) + 1):

            y1 = i * block_size
            x1 = j * block_size

            y2 = (i+1) * block_size
            x2 = (j+1) * block_size

            # print('Block {})'.format((i, j)))

            if y2 > out_3d.shape[0]:
                # y2 = out_3d.shape[0]
                continue
            if x2 > out_3d.shape[1]:
                # x2 = out_3d.shape[1]
                continue

            in_block = images_3d[y1:y2, x1:x2, :, :]

            weight_block = W_3d[y1:y2, x1:x2, :]
            # weight_block = np.ones(weight_block.shape, dtype=weight_block.dtype) * 0.5

            out_block = process_cube(
                img=in_block,
                weight=weight_block**2,
                quality=30
            )

            out_3d[y1:y2, x1:x2, :, :] = out_block

            # plt.figure()
            # plt.subplot(211), plt.imshow(in_block[:,:,:,0])
            # plt.title('in_block')
            # plt.subplot(212), plt.imshow(out_block[:,:,:,0])
            # plt.title('out_block')
            #
            # plt.figure()
            # plt.imshow(images_3d[:, :, :, 0])
            # plt.title('input image 0')
            #
            # plt.figure()
            # plt.imshow(left_images[0][:, :, :])
            # plt.title('input image 0, just left image directly')
            #
            # plt.figure()
            # plt.imshow(out_3d[:, :, :, 0])
            # plt.title('output image 0')
            #
            # plt.show()
            #
            # import sys
            # sys.exit(0)

    # out_3d = out_3d[0:int(out_3d.shape[0]/block_size)*block_size, 0:int(out_3d.shape[1]/block_size)*block_size, :, :]

    out_images = []
    for i in range(block_size):
        out_images.append(out_3d[:, :, :, i])

    return out_images


def process(left_image_paths, right_image_paths, out_dir):

    N = len(left_image_paths)
    print(len(left_image_paths))
    print(len(right_image_paths))
    assert(len(left_image_paths) == len(right_image_paths))

    img0 = cv2.imread(left_image_paths[0], cv2.IMREAD_COLOR)
    frame_size = (img0.shape[1], img0.shape[0])

    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # out = cv2.VideoWriter(
    #     filename=str(out_dir / 'output.avi'),
    #     fourcc=fourcc,
    #     fps=fps,
    #     frameSize=frame_size
    # )
    #
    # out_orig = cv2.VideoWriter(
    #     filename=str(out_dir / 'output_orig.avi'),
    #     fourcc=fourcc,
    #     fps=fps,
    #     frameSize=frame_size
    # )

    # N = 64

    for i in range(N):

        print(i)

        left_image = cv2.imread(left_image_paths[i], cv2.IMREAD_COLOR)
        right_image = cv2.imread(right_image_paths[i], cv2.IMREAD_COLOR)

        out_image = np.uint8(process_image(left_image, right_image))

        out_path = str(out_dir / left_image_paths[i][-14:])
        cv2.imwrite(out_path, out_image)

        # out.write(out_image)
        # out_orig.write(left_image)
        #
        # cv2.imshow('video', out_image)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # block_size = 8
    #
    # for block in range(int(N/block_size)):
    #
    #     left_images = []
    #     right_images = []
    #     image_names = []
    #
    #     for j in range(block_size):
    #
    #         i = block * block_size + j
    #
    #         print('i = {}, processing image {}'.format(i, left_image_paths[i][-14:]))
    #
    #         left_image = cv2.imread(left_image_paths[i], cv2.IMREAD_COLOR)
    #         right_image = cv2.imread(right_image_paths[i], cv2.IMREAD_COLOR)
    #
    #         left_images.append(left_image)
    #         right_images.append(right_image)
    #
    #         image_names.append(left_image_paths[i][-14:])
    #
    #         # out_orig.write(left_image)
    #
    #     out_images = process_image_block(left_images, right_images)
    #
    #     for out_image, image_path in zip(out_images, image_names):
    #
    #         out_image = np.clip(out_image, 0, 255)
    #         out_image = np.uint8(out_image)
    #
    #         out_path = str(out_dir / image_path)
    #         cv2.imwrite(out_path, out_image)
    #
    #         # out.write(out_image)
    #
    #         # cv2.imshow('video', out_image)
    #         # if cv2.waitKey(1) & 0xFF == ord('q'):
    #         #     break
    #
    #         out_images.append(out_image)

    # out.release()
    # out_orig.release()
    # cv2.destroyAllWindows()


def main():

    from pathlib import Path

    p = Path(kitti_path)

    left_image_paths = list(str(x) for x in (p / 'left').glob('*.png'))
    left_image_paths.sort()
    print('Compressing {} images'.format(len(left_image_paths)))

    right_image_paths = list(str(x) for x in (p / 'right').glob('*.png'))
    right_image_paths.sort()

    out_path = p / 'justdct_quality20'

    num_images = 5

    process(left_image_paths[:num_images], right_image_paths[:num_images], out_path)


if __name__ == '__main__':
    main()
