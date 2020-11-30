#!/home/wli/env python3
# -*- coding: utf-8 -*-
"""
Title: task_id_makeup
=====================
Created: 10-31-2019
Python-Version: 3.5

Description:
------------
This module is only used for making up the failed tasks on HPC.

Inputs:
*******

    slide_category = slide_categories[1] : normal slides or tumor slides;

    stride = 56 : the pixels skipped by slide window, for example: 16;

    color_norm = False : if color normalization will be used; 
       color_norm_method = color_norm_methods[0]
       template_image_path = '/home/wli/DeepLearningCamelyon/dldp/data/tumor_st.png'
       log_path = '/raidb/wli/testing_1219/Pred_Heatmap/log_files'

    IIIdhistech_only = False : if only the slides from 3D histech scanner

Output:
*******

    path_for_results


Request:
--------
This module needs the results from task_id_remain.
"""
import numpy as np
import pandas as pd
import sys
import os
import os.path as osp

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
import stain_utils as utils
import stainNorm_Reinhard
import stainNorm_Macenko
import stainNorm_Vahadane


from datetime import datetime
import Pred_Slide_Window_For_Heatmap as pswh


if __name__ == '__main__':

    taskid = int(os.environ['SGE_TASK_ID'])
    # path ='/scratch/weizhe.li/Pred_Storage/test_1003_Macenko_right'
    # preded_taskid = np.load('/home/weizhe.li/makeuptask/test_II_I_real.npy')
    preded_taskid = np.load(
        '/home/weizhe.li/makeuptask/makeup_normal_Macenko_1119_2019-11-19_09:37:33.npy')
    color_norm = True
    # make sure the taskid will not go beyond the limit
    if taskid > len(preded_taskid):
        sys.exit("out of range, program will stop")
    # for test images
    if len(preded_taskid) < 150000:
        taskid = (taskid-1)
    else:
        sys.exit(
            "This is for makeup prediction. Please do the actual prediction first.")

    # setup corresponding taskid

    task_id = preded_taskid[taskid]
    # modify task_id and decide the number of patches to be predicted for one task id from HPC
    # task_id, patches_per_task = pswh.modify_task_id(origin_taskid, slide_category)
    patches_per_task = {'tumor': 180,
                        'normal': 200,
                        'test': 160
                        }
    current_time = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
    slide_categories = ['normal', 'tumor', 'test']
    slide_category = slide_categories[0]
    patches_per_task = patches_per_task[slide_category]
    batch_size = 32
    crop_size = [224, 224]
    pred_size = 224
    NUM_CLASSES = 2  # not_tumor, tumor
    stride = 56
    #origin_taskid = int(os.environ['SGE_TASK_ID'])
    color_norm_methods = ['Vahadane', 'Reinhard', 'Macenko']
    template_image_path = '/home/weizhe.li/tumor_st.png'

    if color_norm:
        color_norm_method = color_norm_methods[2]
        fit = pswh.color_normalization(template_image_path, color_norm_method)
    else:
        color_norm_method = 'baseline'
        fit = None
    # the makup results will have a folder starting with 'makeup'
    path_for_results = '/scratch/weizhe.li/Pred_Storage/makeup_%s_%s' % (
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
    ####################################################################################################
    # the slide and dimension information retrievaled based on the name of index_paths to make sure all
    # dimension, index_paths, slide are all matched
    ####################################################################################################
    index_paths = {
        "normal": '/home/weizhe.li/li-code4hpc/pred_dim_0314/training-updated/normal/patch_index',
        "tumor": '/home/weizhe.li/li-code4hpc/pred_dim_0314/training-updated/tumor/patch_index',
        "test": '/home/weizhe.li/li-code4hpc/pred_dim_0314/testing/patch_index'
    }

    # the slide and dimension information retrievaled based on the name of index_paths to make sure all
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
        '/home/weizhe.li/Trained_Models/googlenetv1_color_norm_r_m_10.10.19 01:59_Macenko-04-0.89.hdf5')
    # '/home/weizhe.li/Trained_Models/googlenetmainmodel_09.23.19 02:24_3dhistech_Baseline-03-0.92.hdf5')

    # identify the slide and patches index
    i, j, j_dif = pswh.slide_patch_index(
        task_id, patches_per_task, patch_number)
    # select slides from 3dhistech scanner
    if IIIdhistech_only:

        pswh.exit_program(i, slide_category)

    all_samples, n_samples, slide, new_slide_path = pswh.slide_level_param(
        i, index_path_pred, slide_path_pred)

    path_to_create = pswh.creat_folder(new_slide_path, path_for_results)

    sub_samples, range_right = pswh.patches_for_pred(
        i, j, j_dif, patch_number, patches_per_task, all_samples)

    pswh.batch_pred_per_taskid(pred_size, stride, sub_samples, slide, fit, model, range_right,
                               path_to_create, task_id, patch_number, i, j, current_time, log_file, color_norm=False)
