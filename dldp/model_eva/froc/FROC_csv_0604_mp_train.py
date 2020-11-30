#! /home/wli/env python3
# -*- coding: utf-8 -*-
"""
Title: lesion based prediction
==============================
Created: 10-31-2019
Python version: 3.5/3.6

Description:
------------
    This module is used to extract the location (coordinates) and
    probability of individual predicted lesions from heatmaps.

    Input:  heatmap (two heatmaps, one is for generating binary image;
            the other one is to be combined with previous one to get probability score.
            dimension file
            Please change the index number of three lists: label_heatmap_list, value_heatmap_list, dim_path_list.
    Output: CSV file for each slide
            Please set the output folder to store the results.
    
Note:
-----

This module should be run before the module generating FROC curve.
"""
import csv
import glob
import os
import random

import cv2
import numpy as np
import scipy.stats.stats as st
from skimage.measure import label
from skimage.measure import regionprops
from skimage.segmentation import clear_border
from skimage.morphology import closing, square
from matplotlib import cm
import matplotlib
from tqdm import tqdm
from skimage.filters import threshold_otsu
from keras.models import load_model
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
from keras.utils.np_utils import to_categorical

import os.path as osp
import openslide
from pathlib import Path
import numpy as np
import skimage.io as io
import skimage.transform as trans
import re
import scipy.ndimage as nd
import sys
from multiprocessing import Process
from multiprocessing import Pool
import logging
sys.path.append('/home/weizhe.li/dldp/utils/logman')
sys.path.append('/home/weizhe.li/dldp/utils')
# setup_logging

import fileman as fm
import logger_management
from logger_management import log_with_template
from logger_management import StreamToLogger
from logger_management import setup_logging


def get_region_props_kernel(heatmapbinary, heatmap, kernel_size):
    """
    Get properties of heatmap
    :param heatmapbinary: binarized heatmap according to a set threshold.
    :type heatmapbinary: binary array
    :param heatmap: the probability map
    :type heatmap: array
    :return: image property
    :rtype: object, the return from a function "regionprops" from
            skimage.measure package
    """
    # heatmapbinary = closing(heatmapbinary, square[3])
    # heatmapbinary = clear_border(heatmapbinary)
    # open_heatmapbinary = nd.binary_opening(heatmapbinary)
    # close_heatmapbinary = nd.binary_closing(open_heatmapbinary)
    #heatmap_binary = heatmapbinary
    heatmap_binary = heatmapbinary.astype("uint8")
    #filled_image = nd.morphology.binary_fill_holes(heatmap_binary)

    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)) # 10-29-19 using the original kernel to compare.
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)) # 10-31-19 using the  kernel 20 to compare.
    # 10-31-19 using the  kernel 20 to compare.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    #heatmap_close = cv2.morphologyEx(heatmap_binary, cv2.MORPH_CLOSE, kernel)
    #heatmap_open = cv2.morphologyEx(heatmap_close, cv2.MORPH_OPEN, kernel)
    heatmap_open = cv2.morphologyEx(heatmap_binary, cv2.MORPH_OPEN, kernel)
    heatmap_close = cv2.morphologyEx(heatmap_open, cv2.MORPH_CLOSE, kernel)
    #matplotlib.image.imsave('%s/%s.png' % (result_folder, label_heatmap_paths[i].split('/')[-3]),
    #                        heatmap_open)
    #labeled_img = label(heatmap_open)
    matplotlib.image.imsave('%s/%s.png' % (result_folder, label_heatmap_paths[i].split('/')[-3]),
                            heatmap_close)
    labeled_img = label(heatmap_close)
    return regionprops(labeled_img, intensity_image=heatmap)


def get_region_props_binary_fill(heatmapbinary, heatmap):
    """
    The function is to get location and probability of heatmap
    using ndimage.

    :param heatmapbinary: the binarized heatmap
    :type heatmap_binary: array
    :param heatmap: the original heat_map
    :type heatmap: array
    :returns: regionprops
    :rtype: object

    """
    # heatmapbinary = closing(heatmapbinary, square[3])
    # heatmapbinary = clear_border(heatmapbinary)
    # open_heatmapbinary = nd.binary_opening(heatmapbinary)
    # close_heatmapbinary = nd.binary_closing(open_heatmapbinary)
    heatmap_binary = heatmapbinary.astype("uint8")
    filled_image = nd.morphology.binary_fill_holes(heatmap_binary)

    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    #heatmap_close = cv2.morphologyEx(heatmap_binary, cv2.MORPH_CLOSE, kernel)
    #heatmap_open = cv2.morphologyEx(heatmap_close, cv2.MORPH_OPEN, kernel)

    matplotlib.image.imsave('%s/%s.png' % (result_folder, label_heatmap_paths[i].split('/')[-3]),
                            filled_image)
    labeled_img = label(filled_image, connectivity=2)
    return regionprops(labeled_img, intensity_image=heatmap)


def list_heatmap_file_in_dir(path):
    """
    The function is used to return a list of files in a specific directory and
    its subdirectories.
    :param path: the interested directory
    :type path: str
    :param file_ext: file extension. for exaple, 'tif', 'jpg'
    :type file_ext: str
    :return: a list of files with their absolute paths
    :rtype: list
    """

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
#           if 'pred[].%s' % file_ext in file:
            if re.search(r'preds_oldmodel_.*\.npy', file):
                files.append(os.path.join(r, file))
    files.sort()
    return files

def lesion_coord(i):

    label_heatmap = np.load(label_heatmap_paths[i])
    new_slide_path = [x for x in value_heatmap_paths if re.search(label_heatmap_paths[i].split('/')[-3], x)]
    value_heatmap_path  = new_slide_path[0]
    value_heatmap = np.load(value_heatmap_path)    
    
    froc_csv = []

    col = ['probability', 'x_coordinate', 'y_coordinate']
    new_dim_path = [x for x in dim_paths if re.search(label_heatmap_paths[i].split('/')[-3], x)]
    dim_path = new_dim_path[0]
    print(dim_path)
    dim = np.load(dim_path)
    dim_x = dim[3]
    dim_y = dim[5]
    # Set a threshold to 0.9
    heatmapbinary_lesion = (label_heatmap > 0.9) * 1
    # Get the average of two heatmaps
    ave_heatmap = (np.array(label_heatmap) + np.array(value_heatmap))/2
    # heatmapbinary_lesion = (heatmap > 0.5)*1

    region_props = get_region_props_kernel(
        heatmapbinary_lesion, ave_heatmap, kernel_size)
    number_lesion = len(region_props)

    # create an empty dataframe to store the region probability and coordinates

    for index in range(number_lesion):
        # the region_props has "index" for individual lesions; the second field is the measurement for property.
        [x, y] = region_props[index]['centroid']
        probability_ave = region_props[index]['mean_intensity']
        #prob_coord = [probability_ave, y * 16 * 4 + dim_x*224,
        #              x * 16 * 4 + dim_y*224]  # 10-29-2019: for reduced map
        prob_coord = [probability_ave, y * 16 + dim_x*224, x * 16 + dim_y*224]
        #prob_coord = [probability_ave, x * 16 + dim_x*224, y * 16 + dim_y*224]
        #prob_coord = [probability_ave, y * 16 + dim_x, x * 16 + dim_y]
        print(prob_coord)
        froc_csv.append(prob_coord)
        #        froc_csv.append(pd.Series(prob_coord, index=(
        #            ['probability', 'x_coordinate', 'y_coordinate'])), ignore_index=True)
        # load the prediction for test data set if a image is normal, the dataframe will be kept empty; if is tumor, will continue to get coordinates.

        #    test_data = np.load(
        #        '~/data_sheet_for_random_forest_16_strike_test_09_pred.csv')
        #
        #    if test_data['is_tumor'] == 0:
        #
        #        froc_csv = froc_csv
        #
        #    else:
        #
        #        for index in range(number_lesion):
        #            [x, y] = region_props[index]['centroidarray']
        #            probability_ave = region_props[index]['mean_intensity']
        #            prob_coord = [probability_ave, y*16 + dim_x, x*16 + dim_y]
        #
        #            froc_csv.append(pd.Series(prob_coord, index=(
        #                ['probability', 'x_coordinate', 'y_coordinate'])), ignore_index=True)
        # Convert the list to a dataframe
    froc_csv_data = pd.DataFrame(froc_csv, columns=col)
    # Save to CSV file with the name same as its slide name.
    froc_csv_data.to_csv('%s/%s.csv' % (result_folder, label_heatmap_paths[i].split('/')[-3]))


def lesion_coord_with_roc(i, based_on_roc=True):

    label_heatmap = np.load(label_heatmap_paths[i])
    new_slide_path = [x for x in value_heatmap_paths if re.search(label_heatmap_paths[i].split('/')[-3], x)]
    value_heatmap_path  = new_slide_path[0]
    value_heatmap = np.load(value_heatmap_path)
    
    # swap label_heatmap & value_heatmap
    #label_heatmap_old = label_heatmap
    #value_heatmap_old = value_heatmap
    #label_heatmap = value_heatmap_old
    #value_heatmap = label_heatmap_old
        
    froc_csv = []

    col = ['probability', 'x_coordinate', 'y_coordinate']
    if based_on_roc:
        # 11-06-19: read the slide level prediction
        
#       row_index = int(ref[ref['slide_name'].str.find(case[0:-4])!=-1].index[0])
        slide_level_pred = ROC_result['scores_method_II_16_stride_color_norm_9.0'].at[i]
        print(slide_level_pred)

        if slide_level_pred > 0.5:
            # new_slide_path = [x for x in slide_paths if re.search(
            #    osp.basename(heatmap_paths[i]).replace('.npy', '.tif'), x)]
            # slide_path = new_slide_path[0]
            # slide_path = glob.glob(osp.join(slide_path, os.rename(split(basename(heatmap_path[i])))))

            # data_sheet_for_random_forest.at[i, 'name'] = osp.basename(slide_paths[i])
            # Get the location information. The heatmap is only for the tissue region. To find the location of a lesion
            # in a slide, the coordinates of top-left point of the heatmap in a slide is retrieved.
            #new_dim_path = [x for x in index_paths if re.search(
            #    osp.basename(label_heatmap_paths[i]), x)]

            new_dim_path = [x for x in dim_paths if re.search(label_heatmap_paths[i].split('/')[-3], x)]
            dim_path = new_dim_path[0]
            print(dim_path)
            dim = np.load(dim_path)
            dim_x = dim[3]
            dim_y = dim[5]
            # Set a threshold to 0.9
            heatmapbinary_lesion = (label_heatmap > 0.9) * 1
            # Get the average of two heatmaps
            ave_heatmap = (np.array(label_heatmap) +
                            np.array(value_heatmap))/2
            # heatmapbinary_lesion = (heatmap > 0.5)*1

            region_props = get_region_props_kernel(
                heatmapbinary_lesion, ave_heatmap, kernel_size)
            number_lesion = len(region_props)

            # create an empty dataframe to store the region probability and coordinates

            for index in range(number_lesion):
                # the region_props has "index" for individual lesions; the second field is the measurement for property.
                [x, y] = region_props[index]['centroid']
                probability_ave = region_props[index]['mean_intensity']
                # weighted probability based on slide level prediction
                probability_ave = probability_ave*slide_level_pred
                prob_coord = [probability_ave, y * 16 + dim_x*224, x * 16 + dim_y*224]
                #prob_coord = [probability_ave, y * 16 * 4 + dim_x*224,
                #                x * 16 * 4 + dim_y*224]  # 10-29-2019: for reduced map
                #prob_coord = [probability_ave, y * 16 + dim_x*224, x * 16 + dim_y*224]
                #prob_coord = [probability_ave, x * 16 + dim_x*224, y * 16 + dim_y*224]
                #prob_coord = [probability_ave, y * 16 + dim_x, x * 16 + dim_y]
                print(prob_coord)
                froc_csv.append(prob_coord)
    froc_csv_data = pd.DataFrame(froc_csv, columns=col)
    # Save to CSV file with the name same as its slide name.
    froc_csv_data.to_csv('%s/%s.csv' % (result_folder,
                                            label_heatmap_paths[i].split('/')[-3]))

if __name__ == '__main__':

    # The place to put the results
    module_name = sys.modules['__main__'].__file__
    log_template_path = '/home/weizhe.li/dldp/utils/logman/logging_config.yaml'
    log_with_template(log_template_path, module_name)
    kernel_size = 10
    # the dir for get binary mask
        # heatmap with color normalization
    value_heatmap_path_list = ['/scratch/mikem/UserSupport/weizhe.li/runs_process_cn_V2_True/normal_wnorm_448_400_7694229/',
                         '/scratch/mikem/UserSupport/weizhe.li/runs_process_cn_V2_True/tumor_wnorm_448_400_7694290/',
                         '/scratch/mikem/UserSupport/weizhe.li/runs_process_cn_V2_True/testing_wnorm_448_400_7694222/']
    # heatmap without color normalization
    #list_heatmap_path = ['/scratch/mikem/UserSupport/weizhe.li/runs_process_cn_V2_False/normal_wnorm_448_400_7694348/',
    #                     '/scratch/mikem/UserSupport/weizhe.li/runs_process_cn_V2_False/tumor_wnorm_448_400_7694415/',
    #                     '/scratch/mikem/UserSupport/weizhe.li/runs_process_cn_V2_False/testing_wnorm_448_400_7694088/']
    # heatmap without hnm
    label_heatmap_path_list = ['/projects01/wxc4/wli/CAMELYON16/color_norm_noise_No_HNM/normal_wnorm_448_400_7712446', 
                          '/projects01/wxc4/wli/CAMELYON16/color_norm_noise_No_HNM/tumor_wnorm_448_400_7712321',
                          '/projects01/wxc4/wli/CAMELYON16/color_norm_noise_No_HNM/testing_wnorm_448_400_7713068']

    value_heatmap_paths = list_heatmap_file_in_dir(value_heatmap_path_list[1])
    value_heatmap_paths.sort()
    print('number of value_heatmap: %d' % len(value_heatmap_paths))

    label_heatmap_paths = list_heatmap_file_in_dir(label_heatmap_path_list[1])
    label_heatmap_paths.sort()
    print('number of label_heatmap: %d' % len(label_heatmap_paths))
    print(label_heatmap_paths)

    


    # label_heatmap_path = '/home/wli/Downloads/heat_map/Method_II_Model_I_HNM_no_norm/test_0425_new_new'
    #label_heatmap_path = '/raidb/wli/Final_Results/Heat_map/Method_II/color_noise_color_normalization/Method_II_Model_I_norm/test_0506'
    #label_heatmap_paths = glob.glob(osp.join(label_heatmap_path, '*.npy'))
    #label_heatmap_paths.sort()
    # print(heatmap_paths)

    #value_heatmap_path = '/raidb/wli/Final_Results/Heat_map/Method_II/color_noise_color_normalization/Method_II_Model_II_norm/test_0516'
    #value_heatmap_paths = glob.glob(osp.join(value_heatmap_path, '*.npy'))
    #value_heatmap_paths.sort()
    # index_path = '/Users/liw17/Documents/pred_dim_0314/testing/'

    dim_path_list = ["/home/weizhe.li/li-code4hpc/pred_dim_0314/training-updated/normal/dimensions",
                     "/home/weizhe.li/li-code4hpc/pred_dim_0314/training-updated/tumor/dimensions",
                     "/home/weizhe.li/li-code4hpc/pred_dim_0314/testing/dimensions"]
    dim_paths = glob.glob(osp.join(dim_path_list[1], '*.npy'))
    dim_paths.sort()
    print(dim_paths)

    based_on_roc = False

    if based_on_roc:
        ROC_result = pd.read_csv(
            '/scratch/weizhe.li/RF_trained_models/RF_models_color_norm_repeat/9.0/reference_with_redo_results_16_stride_9.0.csv')

    result_folder = '/projects01/wxc4/wli/CAMELYON16/FROC/tumor_kernel_10_0520'
    try:
        os.makedirs(result_folder)
    except:
        print('skipped')

    for i in range(len(label_heatmap_paths)):
        p = Process(target=lesion_coord, args=(i,))
        p.start()
        
        
        
        
        
     