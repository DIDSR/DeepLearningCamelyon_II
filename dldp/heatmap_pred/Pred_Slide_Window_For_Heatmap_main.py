#!/home/wli/env python3
# -*- coding: utf-8 -*-
"""
Title: Pred_Slide_Window_For_Heatmap_main
=========================================
Created: 10-31-2019
Python-Version: 3.5

Description:
------------
This module is used for slide window based prediction on HPC.

A larger region (448x448 pixels) will be read to avoid edge effect.

Inputs:
*******

    slide_category = slide_categories[1] : normal slides or tumor slides;

    stride = 56 : the pixels skipped by slide window;

    slide_paths: the paths for "normal", "tumor", and "test" WSIs;

    index_paths: the paths having the information of position of images
                 patches to extracted and predicted.

    patch_numbers: the pathes for the files with the information of 
                   number of patches per slides, which will be used for
                   job allocation for different nodes on HPC or on workstation.


    slide_category = slide_categories[1] : normal slides or tumor slides;

    stride = 56 : the pixels skipped by slide window;

    color_norm = False : if color normalization will be used; 
       color_norm_method = color_norm_methods[0]
       template_image_path = '/home/wli/DeepLearningCamelyon/dldp/data/tumor_st.png'
       log_path = '/raidb/wli/testing_1219/Pred_Heatmap/log_files'

    IIIdhistech_only = False : if only the slides from 3D histech scanner

Output:
*******

    path_for_results = '/raidb/wli/testing_1219/Pred_Heatmap/%s_%s'


Request:
--------
This module needs the library module, Pred_Slide_Window_For_Heatmap.

A shell script file is neccessory to run the python code.

Note:
-----
patch_index.py in utils needs to be run first to get the index_paths.

"""
import Pred_Slide_Window_For_Heatmap as pswh
from datetime import datetime
import stainNorm_Vahadane
import stainNorm_Macenko
import stainNorm_Reinhard
import stain_utils as utils
from matplotlib import cm
from tqdm import tqdm
from skimage.filters import threshold_otsu
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
import openslide
from pathlib import Path
from skimage.filters import threshold_otsu
import glob
import math
# before importing HDFStore, make sure 'tables' is installed by pip3 install tables
from pandas import HDFStore
from openslide.deepzoom import DeepZoomGenerator
from sklearn.model_selection import StratifiedShuffleSplit
import cv2
from keras.utils.np_utils import to_categorical

import os.path as osp
import os
import openslide
from pathlib import Path
from keras.models import Sequential
from keras.layers import Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.models import model_from_json
import numpy as np
import sys
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import re
import staintools
#############################################
import h5py
from keras.utils import HDF5Matrix
sys.path.append('/home/wli/Stain_Normalization-master/')


if __name__ == "__main__":

    current_time = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
    slide_categories = ['normal', 'tumor', 'test']
    slide_category = slide_categories[1]
    batch_size = 32
    crop_size = [224, 224]
    pred_size = 224
    NUM_CLASSES = 2  # not_tumor, tumor
    stride = 56
    origin_taskid = int(os.environ['SGE_TASK_ID'])
    color_norm_methods = ['Vahadane', 'Reinhard', 'Macenko']
    color_norm_method = color_norm_methods[1]
    template_image_path = '/home/weizhe.li/tumor_st.png'
    path_for_results = '/scratch/weizhe.li/Pred_Storage/%s_%s' % (
        slide_category, color_norm_method)
    log_path = '/home/weizhe.li/log_files'
    log_file = open('%s/%s.txt' % (log_path, color_norm_method), 'w')
    IIIdhistech_only = True
    # the WSI images from Camelyon16 challenge.
    slide_paths = {
        "normal": '/scratch/wxc4/CAMELYON16-training/normal/',
        "tumor": '/scratch/wxc4/CAMELYON16-training/tumor/',
        "test": '/scratch/wxc4/CAMELYON16-testing/'
    }

    # the index_path is place to store all the coordinate of tiled patches
    # ####################################################################
    # the slide and dimension information retrievaled based on the name of
    # index_paths to make sure all
    # dimension, index_paths, slide are all matched
    # ####################################################################
    index_paths = {
        "normal": '/home/weizhe.li/li-code4hpc/pred_dim_0314/training-updated/normal/patch_index',
        "tumor": '/home/weizhe.li/li-code4hpc/pred_dim_0314/training-updated/tumor/patch_index',
        "test": '/home/weizhe.li/li-code4hpc/pred_dim_0314/testing/patch_index'
    }

    # the slide and dimension information retrievaled based on the name of
    # index_paths to make sure all
    # dimension, index_paths, slide are all matched
    patch_numbers = {

        "normal": '/home/weizhe.li/PatchNumberForHPC_normal.pkl',
        "tumor": '/home/weizhe.li/PatchNumberForHPC_tumor.pkl',
        "test": '/home/weizhe.li/PatchNumberForHPC_test0314.pkl'
    }
    # collect all the information
    slide_path_pred = pswh.list_file_in_dir(slide_paths[slide_category], 'tif')
    index_path_pred = pswh.list_file_in_dir(index_paths[slide_category], 'pkl')
    patch_number = pd.read_pickle(patch_numbers[slide_category])

    print(slide_path_pred)
    # load the model for prediction
    model = load_model(
        # '/home/wli/Downloads/googlenet0917-02-0.93.hdf5')
        # '/home/wli/googlenetmainmodel1119HNM-02-0.92.hdf5')
        # '/home/weizhe.li/li-code4hpc/GoogleNetV1_01_28HNM_BoarderPatches-01-0.94.hdf5')
        # '/home/weizhe.li/Training/googlenetmainmodel0227HNM_boarderpatches_noise2-00-0.94.hdf5')
        # '/home/weizhe.li/Training/googlenetmainmodel0227HNM_noise-01-0.93.hdf5')
        # '/home/weizhe.li/Training/googlenetmainmodel0430HNM_noise-04-0.92.hdf5')#Method_II_Model_I_color_noise_and_color_norm
        # '/home/weizhe.li/Training/googlenetmainmodel0512HNM_noise_boarder-01-0.92.hdf5') #Method_II_Model_II_color_noise_and_color_norm
        # '/home/weizhe.li/Trained_Models/Reinhard/googlenetmainmodel0923_3dhistech_reinhard-02-0.90.hdf5') #3dhistech_reinhard
        '/home/weizhe.li/Trained_Models/Macenko/googlenetmainmodel0826_3dhistech_macenko-02-0.90.hdf5')
    # modify task_id and decide the number of patches to be predicted for one
    # task id from HPC
    task_id, patches_per_task = pswh.modify_task_id(origin_taskid,
                                                    slide_category)
    # identify the slide and patches index
    i, j, j_dif = pswh.slide_patch_index(task_id, patches_per_task,
                                         patch_number)
    # select slides from 3dhistech scanner
    if IIIdhistech_only:
        pswh.exit_program(i, slide_category)

    all_samples, n_samples, slide, new_slide_path = pswh.slide_level_param(i,
                                                                           index_path_pred,
                                                                           slide_path_pred)

    path_to_create = pswh.creat_folder(new_slide_path, path_for_results)

    sub_samples, range_right = pswh.patches_for_pred(i, j, j_dif,
                                                     patch_number,
                                                     patches_per_task,
                                                     all_samples)

    fit = pswh.color_normalization(template_image_path, color_norm_method)

    pswh.batch_pred_per_taskid(pred_size, stride, sub_samples, slide, fit,
                               model, range_right, path_to_create, task_id,
                               patch_number, i, j, current_time, log_file)
