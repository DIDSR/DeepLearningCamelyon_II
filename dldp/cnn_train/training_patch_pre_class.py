# !/home/wli/env python3
# -*- coding: utf-8 -*-
"""
Title: training patch prepare
=============================

Created: 10-31-2019
Python-Version: 3.5, 3.6
Status: under development

Description:
------------

This libray module provides functions for get training patches ready for
training neural network.

"""

from datetime import datetime
import time
from sklearn.utils import resample
from keras.utils.np_utils import to_categorical
import math
from skimage import io
import cv2 as cv2
from sklearn.model_selection import StratifiedShuffleSplit
from openslide.deepzoom import DeepZoomGenerator
from pandas import HDFStore
import os
import glob
from skimage.filters import threshold_otsu
from pathlib import Path
import openslide
import os.path as osp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
# before importing HDFStore, make sure 'tables' is installed by pip3 install tables
# from scipy.misc import imsave as saveim


def creat_folder(new_slide_path, path_for_results):
    """
    The function is used to create folder for each WSI slide to store the extracted
    patches.

    :param new_slide_path: the folder to be created
    :type new_slide_path: string
    :param path_for_results: the folder to store all the extracted patches
    :type path_for_results: string
    :return: the obsolute path of the new created folder
    :rtype: string

    """
    folder_name = osp.splitext(osp.basename(new_slide_path[0]))[0]
    path_to_create = osp.join(path_for_results, folder_name)

    try:
        os.makedirs(path_to_create)
    except Exception:
        print('Folder exists. Skipped')

    return path_to_create


class patch_pre_for_train(object):

    """
    The class is used to send series of image patches for network training.

    :param image_patch_dir: the folder storing all the image patches including
                            training and validation patches
    :type image_patch_dir: string
    :param exclude_normal_list: the normal slides with partial annotations for
                                tumor region
    :type exclude_normal_list: list
    :param validation_slides_normal: the normal WSI slides for validation
    :type validation_patches_normal: list
    :param validation_slides_tumor: the tumor WSI slides for validation
    :type validation_patches_tumor: list
    :param path_to_save_model: the folder storing the results
    :type path_to_save_model: string
    :param crop_size: the size of image patches sent to network training
    :type crop_size: list
    :param batch_size: the number of image patches per batch
    :type batch_size: int
    :param IIIdhistech_only: only the slides from 3dhistech scanners will be used
    :type IIIdhistech_only: boolean


    :ivar sample_type: "training" or "validation"
    :type sample_type: string

    """

    def __init__(self, image_patch_dir, exclude_normal_list,
                 validation_slides_normal, validation_slides_tumor,
                 path_to_save_model, crop_size, batch_size,
                 IIIdhistech_only=False):

        self.image_patch_dir = image_patch_dir
        self.exclude_normal_list = exclude_normal_list
        self.validation_slides_normal = validation_slides_normal
        self.validation_slides_tumor = validation_slides_tumor
        self.path_to_save_model = path_to_save_model
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.IIIdhistech_only = IIIdhistech_only

    def __patch_finder(self):
        """
        find all the image patches in a folder

        :returns: two dataframes, one for tumor patches; one for normal patches
        :rtype: tuple

        """
        tumor_patches = pd.DataFrame(columns=['patch_path', 'is_tumor'])
        normal_patches = pd.DataFrame(columns=['patch_path', 'is_tumor'])
        tumor_patch_paths_list = []
        normal_patch_paths_list = []

        for folder in os.listdir(self.image_patch_dir):

            if osp.basename(folder).startswith('tumor'):
                for subfolder in os.listdir(osp.join(self.image_patch_dir, folder)):
                    # print(subfolder)

                    tumor_patch_paths_in_folder = glob.glob(
                        osp.join(self.image_patch_dir, folder, subfolder, '*.png'))
                    # print(len(tumor_patch_paths_in_folder))
                    tumor_patch_paths_list.extend(tumor_patch_paths_in_folder)
                    # print(len(tumor_patch_paths_list))

            else:

                for subfolder in os.listdir(osp.join(self.image_patch_dir, folder)):
                    # print(subfolder)
                    normal_patch_paths_in_folder = glob.glob(
                        osp.join(self.image_patch_dir, folder, subfolder, '*.png'))
                    normal_patch_paths_list.extend(
                        normal_patch_paths_in_folder)
                    # print(normal_patch_paths_list)

        print(len(tumor_patch_paths_list))
        tumor_patch_paths_series = pd.Series(tumor_patch_paths_list)
        tumor_patches['patch_path'] = tumor_patch_paths_series.values
        tumor_patches['is_tumor'] = 1
        print(len(tumor_patches))

        normal_patch_paths_series = pd.Series(normal_patch_paths_list)
        normal_patches['patch_path'] = normal_patch_paths_series.values
        normal_patches['is_tumor'] = 0
        print(len(normal_patches))

        return (tumor_patches, normal_patches)

    def training_patch_paths(self):
        """
        generate training and validation patch patches

        :return: training_patches, validation_patches
        :rtype: tuple
        """
        all_patches = self.__patch_finder()
        tumor_patches = all_patches[0]
        print(tumor_patches)
        # print(tumor_patches)

        normal_patches = all_patches[1]
        print(normal_patches)

        # exclude normal patches from not fully annotated tumor slides
        for i in range(len(self.exclude_normal_list)):
            normal_patches = normal_patches[
                ~normal_patches.patch_path.str.contains(self.exclude_normal_list[i])]

        print(len(normal_patches))

        # oversample the tumor patches

        tumor_patches = resample(tumor_patches, replace=True,
                                 n_samples=len(normal_patches), random_state=123)

        # get the time stamp

        time_of_saving = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        tumor_patches.to_csv('%s/tumor_patches_%s.csv' %
                             (self.path_to_save_model, time_of_saving))

        ###########################################################################
        # separate training and validation patches for both normal and tumor slides
        ###########################################################################
        training_patches_normal = normal_patches
        validation_patches_normal = pd.DataFrame(
            columns=['patch_path', 'is_tumor'])
        training_patches_tumor = tumor_patches
        validation_patches_tumor = pd.DataFrame(
            columns=['patch_path', 'is_tumor'])

        for i in range(len(self.validation_slides_normal)):
            training_patches_normal = training_patches_normal[
                ~training_patches_normal.patch_path.str.contains(self.validation_slides_normal[i])]

        for i in range(len(self.validation_slides_normal)):
            to_be_append_normal = normal_patches[
                normal_patches.patch_path.str.contains(self.validation_slides_normal[i])]
            validation_patches_normal = validation_patches_normal.append(
                to_be_append_normal, ignore_index=True)

        for i in range(len(self.validation_slides_tumor)):
            training_patches_tumor = training_patches_tumor[
                ~training_patches_tumor.patch_path.str.contains(self.validation_slides_tumor[i])]

        for i in range(len(self.validation_slides_tumor)):
            to_be_append_tumor = tumor_patches[
                tumor_patches.patch_path.str.contains(self.validation_slides_tumor[i])]
            validation_patches_tumor = validation_patches_tumor.append(
                to_be_append_tumor, ignore_index=True)
            # print(to_be_append_tumor)
            # print(validation_patches_tumor)

        # validation_patches_tumor = tumor_patches[~training_patches_tumor.patch_path]
        print(len(training_patches_tumor))
        print(len(training_patches_normal))
        print(len(validation_patches_tumor))
        print(len(validation_patches_normal))

        # ### keep only 3DHISTECH images################################
        if self.IIIdhistech_only:
            training_patches_tumor_3dhistech = training_patches_tumor[training_patches_tumor['patch_path'].map(
                lambda x: int(osp.splitext(osp.basename(x))[0][6:9]) < 71 or int(osp.splitext(osp.basename(x))[0][6:9]) > 100)]

            training_patches_normal_3dhistech = training_patches_normal[training_patches_normal['patch_path'].map(lambda x: int(osp.splitext(osp.basename(x))[0][6:9]) < 71 or int(
                osp.splitext(osp.basename(x))[0][6:9]) > 100 if osp.basename(x).startswith('tumor') else int(osp.splitext(osp.basename(x))[0][7:10]) < 101)]

            validation_patches_tumor_3dhistech = validation_patches_tumor[validation_patches_tumor['patch_path'].map(
                lambda x: int(osp.splitext(osp.basename(x))[0][6:9]) < 71 or int(osp.splitext(osp.basename(x))[0][6:9]) > 100)]

            validation_patches_normal_3dhistech = validation_patches_normal[validation_patches_normal['patch_path'].map(lambda x: int(osp.splitext(osp.basename(
                x))[0][6:9]) < 71 or int(osp.splitext(osp.basename(x))[0][6:9]) > 100 if osp.basename(x).startswith('tumor') else int(osp.splitext(osp.basename(x))[0][7:10]) < 101)]
            # get the number of patches
            print(len(training_patches_tumor_3dhistech))
            print(len(training_patches_normal_3dhistech))
            print(len(validation_patches_tumor_3dhistech))
            print(len(validation_patches_normal_3dhistech))

            # get the patches for training and validation

            training_patches = pd.concat(
                [training_patches_tumor_3dhistech, training_patches_normal_3dhistech])
            validation_patches = pd.concat(
                [validation_patches_tumor_3dhistech, validation_patches_normal_3dhistech])

        else:

            training_patches = pd.concat(
                [training_patches_tumor, training_patches_normal])
            validation_patches = pd.concat(
                [validation_patches_tumor, validation_patches_normal])

        return (training_patches, validation_patches)

    ###############################################################################
    # Here is the generator for each batch of model training
    ###############################################################################

    def patch_aug_flip_rotate_crop(self, image,
                                   image_name=False,
                                   folder_to_save=False):
        """
        the function generates 224 patches from 256 patches

        :param image: the original image patch
        :type image: array
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

        dy, dx = self.crop_size
        x = np.random.randint(0, 256 - dx + 1)
        y = np.random.randint(0, 256 - dy + 1)
        # index = [x, y]
        # cropped_img = (image[x:(x+dx),y:(y+dy),:],rgb_binary[x:(x+dx),y:(y+dy)],
        # mask[x:(x+dx), y:(y+dy)])
        cropped_img = image_rotated[x:(x + dx), y:(y + dy), :]
        # cropped_binary = rgb_binary[x:(x+dx), y:(y+dy)]
        # cropped_mask = mask[x:(x + dx), y:(y + dy)]

    #    saveim('%s/%s_aug.png' % (self.folder_to_save, image_name), cropped_img)

        return cropped_img

    def patch_aug_flip_rotate_crop_1_to_5(self, image, image_name=False,
                                          folder_to_save=False):
        """
        The function generates 224 patches from 256 patches.
        each image patches will have 5 cropped and rotated image patches

        :param image: the original image
        :type image: array
        :param image_name: the name of original image
        :type image_name: string
        :param self.folder_to_save: the folder storing the new images

        :return: image list including 5 new images
        :rtype: list

        """
        image_list = []
        image_rotated1 = np.fliplr(image)
        image_rotated2 = np.rot90(image, 1)
        image_rotated3 = np.rot90(image, 2)
        image_rotated4 = np.rot90(image, 3)
        image_rotated5 = image

        # maskset = [imgmask, maskroted1, maskroted2, maskroted3, maskroted4]
        # imageset = [img_norm, imageroted1, imageroted2, imageroted3, imageroted4]

        dy, dx = self.crop_size
        x = np.random.randint(0, 256 - dx + 1)
        y = np.random.randint(0, 256 - dy + 1)
        # index = [x, y]
        # cropped_img = (image[x:(x+dx),y:(y+dy),:],rgb_binary[x:(x+dx),y:(y+dy)],
        # mask[x:(x+dx), y:(y+dy)])
        cropped_img1 = image_rotated1[x:(x + dx), y:(y + dy), :]
        image_list.append(cropped_img1)
        cropped_img2 = image_rotated2[x:(x + dx), y:(y + dy), :]
        image_list.append(cropped_img2)
        cropped_img3 = image_rotated3[x:(x + dx), y:(y + dy), :]
        image_list.append(cropped_img3)
        cropped_img4 = image_rotated4[x:(x + dx), y:(y + dy), :]
        image_list.append(cropped_img4)
        cropped_img5 = image_rotated5[x:(x + dx), y:(y + dy), :]
        image_list.append(cropped_img5)
        # cropped_binary = rgb_binary[x:(x+dx), y:(y+dy)]
        # cropped_mask = mask[x:(x + dx), y:(y + dy)]

        # print(image_list)
    #    saveim('%s/%s_aug.png' % (self.folder_to_save, image_name), cropped_img)

        return image_list

    @staticmethod
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

    @staticmethod
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

    def gen_imgs(self, sample_type, shuffle=True):
        """
        The a generator used to feed neural network with image patches.

        :param sample_type: "training" or "validation"
        :type sample_type: string
        :param shuffle: if the images will be shuffled or not
        :type shuffle: boolean
        :returns: X_train, y_train
        :rtype: array

        """
        train_val_samples = self.training_patch_paths()
        if sample_type == 'training':
            samples = train_val_samples[0]
        elif sample_type == 'validation':
            samples = train_val_samples[1]

        num_samples = len(samples)
        while 1:
            if shuffle:
                # if frac = 1 will reorganized list randomly
                samples = samples.sample(frac=1)

            for offset in range(0, num_samples, self.batch_size):
                batch_samples = samples.iloc[offset:offset +
                                             self.batch_size]
                images = []
                labels = []
                for _, batch_sample in batch_samples.iterrows():
                    img = io.imread(batch_sample.patch_path)
                    img = img[:, :, :3]
                    image_name = osp.splitext(
                        osp.basename(batch_sample.patch_path))[0]

                    img = self.patch_aug_flip_rotate_crop(
                        img, self.crop_size)
                    label = batch_sample['is_tumor']

                    images.append(np.array(img))
                    labels.append(label)

                X_train = np.array(images)
                y_train = np.array(labels)
                y_train = to_categorical(y_train, num_classes=2)

                yield X_train, y_train

    def gen_imgs_1_to_5(self, sample_type, shuffle=True):
        """
        The a generator used to feed neural network with image patches.
        Each orginal patch will give 5 new image patches after augmentation.

        :param sample_type: "training" or "validation"
        :type sample_type: string
        :param shuffle: if the images will be shuffled or not
        :type shuffle: boolean
        :returns: X_train, y_train
        :rtype: array

        """
        train_val_samples = self.training_patch_paths()
        if sample_type == 'training':
            samples = train_val_samples[0]
        elif sample_type == 'validation':
            samples = train_val_samples[1]

        num_samples = len(samples)
        while 1:
            if shuffle:
                # if frac = 1 will reorganized list randomly
                samples = samples.sample(frac=1)

            for offset in range(0, num_samples, self.batch_size):
                batch_samples = samples.iloc[offset:offset +
                                             self.batch_size]
                images = []
                labels = []
                for _, batch_sample in batch_samples.iterrows():
                    img = io.imread(batch_sample.patch_path)
                    img = img[:, :, :3]
                    image_name = osp.splitext(
                        osp.basename(batch_sample.patch_path))[0]

                    img = self.patch_aug_flip_rotate_crop_1_to_5(
                        img, self.crop_size)
                    # img = patch_aug_flip_rotate_crop(
                    #    img, crop_size, image_name, self.folder_to_save)
                    label = batch_sample['is_tumor']

                    # images.append(img)
                    images.extend(img)
                    labels.append(label)
                    labels.append(label)
                    labels.append(label)
                    labels.append(label)
                    labels.append(label)

                X_train = np.array(images)
                y_train = np.array(labels)
                y_train = to_categorical(y_train, num_classes=2)

                yield X_train, y_train

    def gen_imgs_color_noise(self, sample_type, shuffle=True):
        """
        The a generator used to feed neural network with image patches.

        :param sample_type: "training" or "validation"
        :type sample_type: string
        :param shuffle: if the images will be shuffled or not
        :type shuffle: boolean
        :returns: X_train, y_train
        :rtype: array


        """
        train_val_samples = self.training_patch_paths()
        if sample_type == 'training':
            samples = train_val_samples[0]
        elif sample_type == 'validation':
            samples = train_val_samples[1]

        num_samples = len(samples)
        while 1:
            if shuffle:
                # if frac = 1 will reorganized list randomly
                samples = samples.sample(frac=1)

            for offset in range(0, num_samples, self.batch_size):
                batch_samples = samples.iloc[offset:offset +
                                             self.batch_size]
                images = []
                labels = []
                for _, batch_sample in batch_samples.iterrows():
                    img = io.imread(batch_sample.patch_path)
                    img = img[:, :, :3]
                    image_name = osp.splitext(
                        osp.basename(batch_sample.patch_path))[0]

                    img = self.patch_aug_flip_rotate_crop(
                        img, self.crop_size)
                    img_noise = patch_pre_for_train.color_noise_hsv_to_blue(
                        img)
                    label = batch_sample['is_tumor']

                    images.append(np.array(img))
                    labels.append(label)
                    images.append(np.array(img_noise))
                    labels.append(label)

                X_train = np.array(images)
                y_train = np.array(labels)
                y_train = to_categorical(y_train, num_classes=2)

                yield X_train, y_train
