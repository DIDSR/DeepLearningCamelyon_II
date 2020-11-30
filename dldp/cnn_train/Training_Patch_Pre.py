# !/home/wli/env python3
# -*- coding: utf-8 -*-
"""
Title: training patch prepare
==============================

Created: 10-31-2019
Python-Version: 3.5, 3.6

Description
-----------
This libray module provides functions for get training patches ready for
training neural network.

Note
----
The training patches needed to be extracted before running this code.



"""
import os
#os.environ['KERAS_BACKEND'] = 'theano'
from datetime import datetime
from sklearn.utils import resample
from keras.utils.np_utils import to_categorical
from skimage import io
import cv2 as cv2
import glob
import os.path as osp
import pandas as pd
import numpy as np
# from scipy.misc import imsave as saveim
from imageio import imwrite as saveim
import sys


from dldp.utils import fileman as fm


def creat_folder(new_slide_path, path_for_results):
    """
    To create folder to store the prediction results.
    :param new_slide_path: slide path
    :type new_slide_path: string
    :param path_for_results: the folder storing results
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


def patch_finder(image_patch_dir, hnm_dir, pnt_dir):
    """
    list all the training patches and put their information into a dataframe.

    :param image_patch_dir: the folder storing all the image patches including
                            training and validation patches
    :type image_patch_dir: string
    :returns: two dataframes, one for tumor patches; one for normal patches
    :rtype: tuple

    """
    tumor_patches = pd.DataFrame(columns=['patch_path', 'is_tumor'])
    normal_patches = pd.DataFrame(columns=['patch_path', 'is_tumor'])
    tumor_patch_paths_list = []
    normal_patch_paths_list = []

    for folder in os.listdir(image_patch_dir):

        if osp.basename(folder).startswith('tumor'):
            for subfolder in os.listdir(osp.join(image_patch_dir, folder)):
                # print(subfolder)

                tumor_patch_paths_in_folder = glob.glob(
                    osp.join(image_patch_dir, folder, subfolder, '*.png'))
                # print(len(tumor_patch_paths_in_folder))
                tumor_patch_paths_list.extend(tumor_patch_paths_in_folder)
                # print(len(tumor_patch_paths_list))

        else:

            for subfolder in os.listdir(osp.join(image_patch_dir, folder)):
                # print(subfolder)
                normal_patch_paths_in_folder = glob.glob(
                    osp.join(image_patch_dir, folder, subfolder, '*.png'))
                normal_patch_paths_list.extend(normal_patch_paths_in_folder)
                # print(normal_patch_paths_list)

    # print(len(tumor_patch_paths_list))
    tumor_patch_paths_series = pd.Series(tumor_patch_paths_list)
    tumor_patches['patch_path'] = tumor_patch_paths_series.values
    tumor_patches['is_tumor'] = 1
    # print(len(tumor_patches))

    normal_patch_paths_series = pd.Series(normal_patch_paths_list)
    normal_patches['patch_path'] = normal_patch_paths_series.values
    normal_patches['is_tumor'] = 0
    # print(len(normal_patches))

    hnm_patches_df = pd.DataFrame(columns=['patch_path', 'is_tumor'])
    pnt_patches_df = pd.DataFrame(columns=['patch_path', 'is_tumor'])

    if hnm_dir:

        hnm_patches = fm.list_file_in_dir_II(hnm_dir, 'png')
        hnm_patches_series = pd.Series(hnm_patches)
        hnm_patches_df['patch_path'] = hnm_patches_series.values
        hnm_patches_df['is_tumor'] = 0

    if pnt_dir:

        pnt_patches = fm.list_file_in_dir_II(pnt_dir, 'png')
        pnt_patches_series = pd.Series(pnt_patches)
        pnt_patches_df['patch_path'] = pnt_patches_series.values
        pnt_patches_df['is_tumor'] = 0
        # the normal_patches here will include hnm_patches_df if hnm_dir is True

    return (tumor_patches, normal_patches, hnm_patches_df, pnt_patches_df)


def training_patch_paths(image_patch_dir, exclude_normal_list,
                         validation_slides_normal, validation_slides_tumor,
                         path_to_save_model, IIIdhistech_only, hnm_dir, pnt_dir):
    """
    generate training and validation patch patches

    :param image_patch_dir: the folder storing all the image patches
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
    :param IIIdhistech_only: only the slides from 3dhistech scanners will be used
    :type IIIdhistech_only: boolean

    :return: training_patches, validation_patches
    :rtype: tuple
    """
    all_patches = patch_finder(image_patch_dir, hnm_dir, pnt_dir)
    tumor_patches = all_patches[0]
    # print(len(tumor_patches))
    # print(tumor_patches)

    normal_patches = all_patches[1]
    # print(normal_patches)

    hnm_patches = all_patches[2]
    pnt_patches = all_patches[3]

    # print(pnt_patches)

    for i in range(len(exclude_normal_list)):
        normal_patches = normal_patches[
            ~normal_patches.patch_path.str.contains(exclude_normal_list[i])]

    # print(number_of_normal)

    # oversample the tumor patches
    np.random.seed(123)
    if not hnm_dir:
        number_of_normal = len(normal_patches)
        tumor_patches = resample(tumor_patches, replace=True,
                              n_samples=number_of_normal, random_state=123)


    ###########################################################################
    # separate training and validation patches for both normal and tumor slides
    ###########################################################################
    training_patches_normal = normal_patches
    # if adding hnm_dir here, hnm_dir will be at validation set.
    # if hnm_dir:
    # training_patches_normal = pd.concat(
    # [training_patches_normal, hnm_patches], ignore_index=True)

    # number_of_normal = len(training_patches_normal)

    # tumor_patches = resample(tumor_patches, replace=True,
    # n_samples=number_of_normal, random_state=123)
    validation_patches_normal = pd.DataFrame(
        columns=['patch_path', 'is_tumor'])
    training_patches_tumor = tumor_patches
    validation_patches_tumor = pd.DataFrame(columns=['patch_path', 'is_tumor'])

    for i in range(len(validation_slides_normal)):
        training_patches_normal = training_patches_normal[
            ~training_patches_normal.patch_path.str.contains(validation_slides_normal[i])]

    for i in range(len(validation_slides_normal)):
        to_be_append_normal = normal_patches[
            normal_patches.patch_path.str.contains(validation_slides_normal[i])]
        validation_patches_normal = validation_patches_normal.append(
            to_be_append_normal, ignore_index=True)

    for i in range(len(validation_slides_tumor)):
        training_patches_tumor = training_patches_tumor[
            ~training_patches_tumor.patch_path.str.contains(validation_slides_tumor[i])]

    for i in range(len(validation_slides_tumor)):
        to_be_append_tumor = tumor_patches[
            tumor_patches.patch_path.str.contains(validation_slides_tumor[i])]
        validation_patches_tumor = validation_patches_tumor.append(
            to_be_append_tumor, ignore_index=True)
        # print(to_be_append_tumor)
        # print(validation_patches_tumor)
    # for i in range(len(validation_slides_normal)):
    #     to_be_append_hnm= hnm_patches[
    #         hnm_patches.patch_path.str.contains(validation_slides_tumor[i])]
    #     validation_patches_hnm_tumor = validation_patches_hnm_tumor.append(
    #         to_be_append_hnm_tumor, ignore_index=True)

    # for i in range(len(validation_slides_tumor)):
    #     to_be_append_hnm_normal = hnm_patches[
    #         hnm_patches.patch_path.str.contains(validation_slides_normal[i])]
    #     validation_patches_hnm_normal = validation_patches_hnm_normal.append(
    #         to_be_append_hnm_normal, ignore_index=True)


    # validation_patches_tumor = tumor_patches[~training_patches_tumor.patch_path]

    
    if hnm_dir:
        hnm_patches_all = hnm_patches.copy()
        hnm_patches = hnm_patches[hnm_patches['patch_path'].map(
            lambda x:  float(osp.splitext(osp.basename(x))[0][-4:]) >= 0.6)]

        print('hnm_patches_before_exclude_validation = %d' % len(hnm_patches))
        for i in range(len(validation_slides_normal)):
            hnm_patches = hnm_patches[
                ~hnm_patches.patch_path.str.contains(validation_slides_normal[i])]

        for i in range(len(validation_slides_tumor)):
            hnm_patches = hnm_patches[
                    ~hnm_patches.patch_path.str.contains(validation_slides_tumor[i])]

        print('hnm_patches = %d' % len(hnm_patches))

        training_patches_normal = pd.concat(
            [training_patches_normal, hnm_patches], ignore_index=True)

        training_patches_tumor = resample(training_patches_tumor, replace=True,
                                          n_samples=len(training_patches_normal), random_state=123)
        
        # validation_patches_hnm = hnm_patches_all[~hnm_patches_all.patch_path.isin(hnm_patches.patch_path)]
        # print('validation_hnm_patches: %d' % len(validation_patches_hnm))

        # validation_patches_normal = pd.concat([validation_patches_normal, validation_patches_hnm], ignore_index=True)
        validation_patches_tumor = resample(validation_patches_tumor, replace=True, n_samples=len(validation_patches_normal), random_state=123)
    if pnt_dir:
        training_patches_normal = pd.concat(
            [training_patches_normal, pnt_patches], ignore_index=True)

    print("The number of tumor patches for training: %d" %
          len(training_patches_tumor))
    print("The number of normal patches for training: %d" %
          len(training_patches_normal))
    print("The number of tumor patches for validation: %d" %
          len(validation_patches_tumor))
    print("The number of normal patches for validation: %d" %
          len(validation_patches_normal))

    time_of_saving = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
    tumor_patches = pd.concat([training_patches_tumor, validation_patches_tumor], ignore_index=False)
    tumor_patches.to_csv('%s/tumor_patches_%s.csv' %
                         (path_to_save_model, time_of_saving))
    # ### keep only 3DHISTECH images################################
    if IIIdhistech_only:
        training_patches_tumor_3dhistech = training_patches_tumor[training_patches_tumor['patch_path'].map(
            lambda x: int(osp.splitext(osp.basename(x))[0][6:9]) < 71 or int(osp.splitext(osp.basename(x))[0][6:9]) > 100)]

        training_patches_normal_3dhistech = training_patches_normal[training_patches_normal['patch_path'].map(
            lambda x: int(osp.splitext(osp.basename(x))[0][6:9]) < 71 or int(
                osp.splitext(osp.basename(x))[0][6:9]) > 100 if osp.basename(x).startswith('tumor') else int(osp.splitext(osp.basename(x))[0][7:10]) < 101)]

        validation_patches_tumor_3dhistech = validation_patches_tumor[validation_patches_tumor['patch_path'].map(
            lambda x: int(osp.splitext(osp.basename(x))[0][6:9]) < 71 or int(osp.splitext(osp.basename(x))[0][6:9]) > 100)]

        validation_patches_normal_3dhistech = validation_patches_normal[validation_patches_normal['patch_path'].map(lambda x: int(osp.splitext(osp.basename(
            x))[0][6:9]) < 71 or int(osp.splitext(osp.basename(x))[0][6:9]) > 100 if osp.basename(x).startswith('tumor') else int(osp.splitext(osp.basename(x))[0][7:10]) < 101)]
        # get the number of patches
        print("The number of tumor patches for training(3D Histech only): %d" %
              len(training_patches_tumor_3dhistech))
        print("The number of normal patches for training(3D Histech only): %d" %
              len(training_patches_normal_3dhistech))
        print("The number of tumor patches for validation(3D Histech only): %d" % len(
            validation_patches_tumor_3dhistech))
        print("The number of normal patches for validation(3D Histech only): %d" % len(
            validation_patches_normal_3dhistech))

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
        # training_patches = training_patches.sample(frac=.2, random_state=123)
        # validation_patches = validation_patches.sample(
        # frac=.2, random_state=123)

    return (training_patches, validation_patches)


###############################################################################
# Here is the generator for each batch of model training
###############################################################################
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
    else:
        cropped_img = image_rotated

    if folder_to_save:

        saveim('%s/%s_aug.png' % (folder_to_save, image_name), cropped_img)

    return cropped_img


def patch_aug_flip_rotate_crop_1_to_5(image, crop_size, image_name,
                                      folder_to_save=False):
    """
    The function generates 224 patches from 256 patches.
    each image patches will have 5 cropped and rotated image patches

    :param image: the original image
    :type image: array
    :param crop_size: the size of new image
    :type crop_size: list
    :param image_name: the name of original image
    :type image_name: string
    :param folder_to_save: the folder storing the new images

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

    dy, dx = crop_size
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
    if folder_to_save:
        for i in range(0, 5):
            saveim('%s/%s_aug.png' % (folder_to_save, image_name),
                   image_list[i])

    return image_list


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

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def preprocess_input_mean(x):
    for c in range(x.shape[-1]):
       mean = np.mean(x[:,:,:, c]) 
       std = np.std(x[:,:,:, c])
       x[:,:,:, c] = (x[:,:,:, c] - mean) / std

    return x

def gen_imgs(samples, batch_size, crop_size, folder_to_save=False, shuffle=True):
    """
    The generator used to feed neural network with image patches.

    :param samples: the total training or validation image patches
    :type samples: dataframe (two columns: path, tumor)
    :param batch_size: the size of batch
    :type batch_size: int
    :param crop_size: the size of cropped images
    :type crop_size: list
    :param folder_to_save: the place to storing the augmented images
    :type folder_to_save: string
    :param shuffle: if the images will be shuffled or not
    :type shuffle: boolean
    :returns: X_train, y_train
    :rtype: array

    """

    num_samples = len(samples)
    while 1:
        if shuffle:
            # if frac = 1 will reorganized list randomly
            samples = samples.sample(frac=1, random_state=123)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset+batch_size]
            images = []
            labels = []
            for _, batch_sample in batch_samples.iterrows():
                img = io.imread(batch_sample.patch_path)
                img = img[:, :, :3]
                image_name = osp.splitext(
                    osp.basename(batch_sample.patch_path))[0]

                img = patch_aug_flip_rotate_crop(
                    img, crop_size, image_name, folder_to_save)
                label = batch_sample['is_tumor']

                images.append(np.array(img))
                labels.append(label)

            X_train = np.array(images)
            X_train = X_train.astype('float')
            # for image value scaling
            X_train = preprocess_input_mean(X_train)
            y_train = np.array(labels)
            y_train = to_categorical(y_train, num_classes=2)

            yield X_train, y_train

def gen_imgs_caffe(samples, batch_size, crop_size, folder_to_save=False, shuffle=True):
    """
    The generator used to feed neural network with image patches.

    :param samples: the total training or validation image patches
    :type samples: dataframe (two columns: path, tumor)
    :param batch_size: the size of batch
    :type batch_size: int
    :param crop_size: the size of cropped images
    :type crop_size: list
    :param folder_to_save: the place to storing the augmented images
    :type folder_to_save: string
    :param shuffle: if the images will be shuffled or not
    :type shuffle: boolean
    :returns: X_train, y_train
    :rtype: array

    """

    num_samples = len(samples)
    while 1:
        if shuffle:
            # if frac = 1 will reorganized list randomly
            samples = samples.sample(frac=1)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset+batch_size]
            images = []
            labels = []
            for _, batch_sample in batch_samples.iterrows():
                img = io.imread(batch_sample.patch_path)
                img = img[:, :, :3]
                image_name = osp.splitext(
                    osp.basename(batch_sample.patch_path))[0]

                img = patch_aug_flip_rotate_crop(
                    img, crop_size, image_name, folder_to_save)
                label = batch_sample['is_tumor']
                #img = img.swapaxes(0,2)
                img = img.transpose((2, 0, 1))
                images.append(np.array(img))
                #labels.append([label, label, label])
                labels.append(label)

            X_train = np.array(images)
            X_train = X_train.astype('float')
            # for image value scaling
            X_train = preprocess_input(X_train)
            y_train = np.array(labels)
            y_train = to_categorical(y_train, num_classes=2)

            yield X_train, [y_train, y_train, y_train]


def gen_imgs_1_to_5(samples, batch_size, crop_size,
                    folder_to_save=False, shuffle=True):
    """
    The generator used to feed neural network with image patches.
    Each orginal patch will give 5 new image patches after augmentation.

    :param samples: the total training or validation image patches
    :type samples: dataframe (two columns: path, tumor)
    :param batch_size: the size of batch
    :type batch_size: int
    :param crop_size: the size of cropped images
    :type crop_size: list
    :param folder_to_save: the place to storing the augmented images
    :type folder_to_save: string
    :param shuffle: if the images will be shuffled or not
    :type shuffle: boolean
    :returns: X_train, y_train
    :rtype: array

    """

    num_samples = len(samples)
    while 1:
        if shuffle:
            # if frac = 1 will reorganized list randomly
            samples = samples.sample(frac=1)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset+batch_size]
            images = []
            labels = []
            for _, batch_sample in batch_samples.iterrows():
                img = io.imread(batch_sample.patch_path)
                img = img[:, :, :3]
                image_name = osp.splitext(
                    osp.basename(batch_sample.patch_path))[0]

                img = patch_aug_flip_rotate_crop_1_to_5(
                    img, crop_size, image_name, folder_to_save)
                # img = patch_aug_flip_rotate_crop(
                #    img, crop_size, image_name, folder_to_save)
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


def gen_imgs_color_noise(samples, batch_size, crop_size,
                         folder_to_save=False, shuffle=True):
    """
    # The generator used to feed neural network with image patches.

    :param samples: the total training or validation image patches
    :type samples: dataframe (two columns: path, tumor)
    :param batch_size: the size of batch
    :type batch_size: int
    :param crop_size: the size of cropped images
    :type crop_size: list
    :param folder_to_save: the place to storing the augmented images
    :type folder_to_save: string
    :param shuffle: if the images will be shuffled or not
    :type shuffle: boolean
    :returns: X_train, y_train
    :rtype: array


    """

    num_samples = len(samples)
    while 1:
        if shuffle:
            # if frac = 1 will reorganized list randomly
            samples = samples.sample(frac=1)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset+batch_size]
            images = []
            labels = []
            for index, batch_sample in batch_samples.iterrows():
                try:
                    img = io.imread(batch_sample.patch_path)
                except:
                    try:
                        img = io.imread(
                            batch_samples.iloc[int(index) - 1]['patch_path'])
                    except:
                        img = io.imread(
                            batch_samples.iloc[int(index) + 1]['patch_path'])
                img = img[:, :, :3]
                image_name = osp.splitext(
                    osp.basename(batch_sample.patch_path))[0]

                img = patch_aug_flip_rotate_crop(
                    img, crop_size, image_name, folder_to_save)
                img_noise = color_noise_hsv_to_red(img)
                #img_noise = color_noise_hsv_to_blue(img)
                label = batch_sample['is_tumor']

                images.append(np.array(img))
                labels.append(label)
                images.append(np.array(img_noise))
                labels.append(label)

            X_train = np.array(images)
            y_train = np.array(labels)
            y_train = to_categorical(y_train, num_classes=2)

            yield X_train, y_train
