#!/home/wli/env python3
# -*- coding: utf-8 -*-
"""
Title: patch aug
================

Created: 10-31-2019
Python-Version: 3.5

Description:
------------

This module includes the functions for image cropping and adding color noises.

These functions are included in Training_Patch_Pre.py

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import glob
import cv2
from imageio import imwrite as saveim

# for 224 images


def random_crop2(image, mask, size_origin, crop_size):
    """
    To get randomly cropped images from original image

    :param image: original image
    :type image: array
    :param mask: the corresponding mask image from ground truth
    :type mask: array
    :param size_origin: the size of original image.
    :param crop_size: the size of image to be cropped.

    :return: cropped image, cropped mask, position information
    :rtype: tuple

    """
    # width, height = slide.level_dimensions[4]
    dy, dx = crop_size
    x = np.random.randint(0, size_origin - dx + 1)
    y = np.random.randint(0, size_origin - dy + 1)
    index = [x, y]
    # cropped_img = (image[x:(x+dx), y:(y+dy),:], rgb_binary[x:(x+dx), y:(y+dy)], mask[x:(x+dx), y:(y+dy)])
    cropped_img = image[x:(x+dx), y:(y+dy), :]
    # cropped_binary = rgb_binary[x:(x+dx), y:(y+dy)]
    cropped_mask = mask[x:(x+dx), y:(y+dy)]

    return (cropped_img, cropped_mask, index)


def patch_aug_flip_rotate_crop(image, crop_size, image_name, folder_to_save=False):
    """
    the function generates 224 patches from 256 patches

    :param image: the original image patch
    :type image: array
    :param crop_size: the sized of new image
    :type crop_size: list
    :param image_name: the name of the original image
    :type image_name: string
    :param folder_to_save: the folder to save the new images
    :type folder_to_save: string

    :return: cropped image
    :rtype: array

    """
    random_number = np.random.randint(0, 4)

    if random_number == 0:
        image_rotated = np.fliplr(image)
    elif random_number == 1:
        image_rotated = np.rot90(image, 1)
    elif random_number == 2:
        image_rotated = np.rot90(image, 2)
    elif random_number == 3:
        image_rotated = np.rot90(image, 3)
    else:
        image_rotated = image

    # maskset = [imgmask, maskroted1, maskroted2, maskroted3, maskroted4]
    # imageset = [img_norm, imageroted1, imageroted2, imageroted3, imageroted4]

    dy, dx = crop_size
    if image.shape[0] > dx:
        x = np.random.randint(0, 256 - dx + 1)
        y = np.random.randint(0, 256 - dy + 1)
        # index = [x, y]
        # cropped_img = (image[x:(x+dx),y:(y+dy),:],rgb_binary[x:(x+dx),y:(y+dy)],
        # mask[x:(x+dx), y:(y+dy)])
        cropped_img = image_rotated[x:(x + dx), y:(y + dy), :]
        # cropped_binary = rgb_binary[x:(x+dx), y:(y+dy)]
        # cropped_mask = mask[x:(x + dx), y:(y + dy)]

    cropped_img = image_rotated

    if folder_to_save:

        saveim('%s/%s_aug.png' % (folder_to_save, image_name), cropped_img)

    return cropped_img


def color_noise_hsv_to_blue(image, max=20):
    """
    adding color noise to the direction of blue based on HSV color space

    :param image: the image to be modified
    :type image: array
    :param max: the range of color noise
    :type max: int

    :return: img_noise_rgb, a RGB image with color noise
    :rtype: array

    """
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_colornoise = img_hsv + np.random.uniform(0, max, size=(1, 3))
    img_colornoise[img_colornoise > 255] = 255
    img_colornoise[:, :, 0][img_colornoise[:, :, 0] > 179] = 179
    img_noise = img_colornoise.astype('uint8')
    img_noise_rgb = cv2.cvtColor(img_noise, cv2.COLOR_HSV2BGR)

    return img_noise_rgb


def color_noise_hsv_to_red(image, max=20):
    """
    This function is used to add color noise to the direction of red
    based on HSV color space.

    :param image: the original image
    :type image: array
    :param max: the range of color noise
    :type max: int

    :return: m_rgb, a RGB image with color noise
    :rtype: array

    """
    m_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    m_colornoise = m_hsv + np.random.uniform(0, max, size=(1, 3))
    # using 'unit8' transformation is dangerous, there are lots of values go
    # beyong 255 after adding random number, the transformation will
    # automatically minus it by 255. So here we set the range for S, V, if >
    # 255, then it is 255; for H, if > 179, then it is 179
    m_colornoise[m_colornoise > 255] = 255
    m_colornoise[:, :, 0][m_colornoise[:, :, 0] > 179] = 179
    # now we can transform the value to unit8 safely
    m_int = m_colornoise.astype('uint8')
    m_rgb = cv2.cvtColor(m_int, cv2.COLOR_HSV2RGB)
    return m_rgb


# this is based on the description of liu yun's paper,"Detecting Cancer Metastases on Gigapixel Pathology Images".
# use the random functions from tensorflow


def color_perturb(image):
    """
    adding color noise and changing the constrast.

    :param image: image to be modified
    :type image: array
    :returns: image with color perturbation
    :rtype: array
    """
    image = tf.image.random_brightness(image, max_delta=64. / 255.)
    image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
    image = tf.image.random_hue(image, max_delta=0.04)
    image = tf.image.random_contrast(image, lower=0.25, upper=1.75)

    return image
