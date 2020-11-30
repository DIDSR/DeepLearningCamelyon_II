#!/home/wli/env python3
# -*- coding: utf-8 -*-
"""
Title: color normalization
==========================
Created: 10-31-2019
Python-Version: 3.5

Description:
------------

This module is used to do the color normalization for image patches.

Input:
template image for color normalization
    template_image_path = '/raidb/wli/tumor_st.png'
 raw image patches:
    source_path = '/home/wli/Documents/original_256_patches'
Output: normalized image patches.

Request: this module relys on:
         https://github.com/wanghao14/Stain_Normalization
         https://github.com/Peter554/StainTools
"""

import dldp.utils.fileman as fm
import numpy as np
from datetime import datetime
from tqdm import tqdm
from imageio import imwrite as saveim
import staintools
from skimage import io
import os
import os.path as osp
import stainNorm_Reinhard
import stainNorm_Macenko
import sys
sys.path.append('/home/wli/Stain_Normalization-master')
# from scipy.misc import imsave as saveim
# import the modules from dldp package


def color_normalization(template_image_path, color_norm_method):
    """
    The function put all the color normalization methods together.

    :param template_image_path: the template image for normalization
    :type template_image_path: string
    :param color_norm_method: the method for color normalization
    :type color_norm_method: string

    :return: color_normalizer. It is the initialized object for the
             actual normalization.
    :rtype: object

    """
    template_image = staintools.read_image(template_image_path)
    standardizer = staintools.LuminosityStandardizer.standardize(
        template_image)
    if color_norm_method == 'Reinhard':
        color_normalizer = stainNorm_Reinhard.Normalizer()
        color_normalizer.fit(standardizer)
    elif color_norm_method == 'Macenko':
        color_normalizer = stainNorm_Macenko.Normalizer()
        color_normalizer.fit(standardizer)
    elif color_norm_method == 'Vahadane':
        color_normalizer = staintools.StainNormalizer(method='vahadane')
        color_normalizer.fit(standardizer)
    return color_normalizer


def color_norm(image_patch, fit, log_file=False):
    """
    To normalize images based on color_normalizer from function color_nor
    malization.
    :param image_patch: the image to be normalized. It can be a path of the image or image itself
    :type image_patch: array
    :param fit: the object of color_normalizer.
    :type fit: object
    :param log_file: the file to record the failed tasks.
    :type log_file: str

    :return: img_norm, the normalized images.
    :rtype: array
    """
    if isinstance(image_patch, str) and osp.isfile(image_patch):
        img = io.imread(image_patch)
    elif isinstance(image_patch, np.ndarray):
        img = image_patch

    img = img[:, :, :3]
    img_norm = []
    try:
        img_standard = staintools.LuminosityStandardizer.standardize(img)
        img_norm = fit.transform(img_standard)
    except Exception as e:
        log_file.write(str(image_patch) + ';' + str(e))
    # print(img_norm)
    return img_norm


def save_color_norm_patches(dest_path, source_path, image_patch, img_norm,
                            color_norm_method):
    """
    The normalized image patches will be saved in the same folder structure
    as the original image patches.
    :param dest_path: the place to store the normalized image patches.
    :type dest_path:string
    :param source_path: the folder to store the original image patches.
    :type source_path: string
    :param file: the full path of the original image patch.
    :type file: string
    :param img_norm: the normalized image patch.
    :type img_norm: array
    :param color_norm_method: the method used for color normalization
    :type color_norm_method: string
    :return: None
    """
    file_full_path = osp.dirname(image_patch)
    relative_path = osp.relpath(file_full_path, source_path)
    path_to_create = osp.join(dest_path, relative_path)
    try:
        os.makedirs(path_to_create)
    except Exception:
        pass
    # print(image_patch)
    try:
        saveim('%s/%s_%s.png' % (path_to_create,
                                 osp.splitext(osp.basename(image_patch))[0],
                                 color_norm_method),
               img_norm)
    except Exception:
        pass

    return None


if __name__ == "__main__":

    template_image_path = '/raidb/wli/tumor_st.png'
    # The template image is also included in the package
    # template_image_path = '.../dldp/data/tumor_st.png'
    source_path = '/home/wli/Documents/original_256_patches'
    # source_path = '/raidb/wli/256_patches_to_be_normalized/original
    # _256_patches'
    dest_path_1 = '/raidb/wli/testing_1219/color_norm/Reinhard'
    dest_path_2 = '/raidb/wli/testing_1219/color_norm/Macenko'
    log_path = '/home/wli/log_files'

    ###########################################################################

    patches_for_color_norm = fm.list_file_in_dir_II(
        source_path, file_ext='png')

    print(len(patches_for_color_norm))
    for color_norm_method in ['Reinhard', 'Macenko']:
        current_time = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        log_file = open('%s/%s_%s.txt' %
                        (log_path, color_norm_method, current_time), 'w')
        fit = color_normalization(template_image_path, color_norm_method)
        if color_norm_method == 'Reinhard':
            dest_path = dest_path_1
        elif color_norm_method == 'Macenko':
            dest_path = dest_path_2
        for image_patch in tqdm(patches_for_color_norm):
            img_norm = color_norm(image_patch, fit, log_file)
            save_color_norm_patches(
                dest_path, source_path, image_patch, img_norm,
                color_norm_method)
