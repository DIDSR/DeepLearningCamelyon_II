#!/home/wli/env python3
# -*- coding: utf-8 -*-
"""
Title: patch index
===========================
Created: 10-31-2019
Python-Version: 3.5, 3.6

Description:
------------
This module is used to generate the x, y coordinates of all the image patches in a WSI.
If an image patch is outside of tissue region, it will be labeled as 0, and will be
assigned to 0 during prediction.

This module is required by hard negative mining and heatmap prediction. 

Input
*****
WSIs

Output
******

a dataframe includes three columns:
            tile_loc: coordinates
            is_tissue: if an image patch is a tissue patch
            slide_path: the path of the WSI

Request
-------

this module relys on:
         OpenSlide: https://github.com/openslide/openslide-python

"""

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
import openslide
from pathlib import Path
from keras.models import Sequential
from keras.layers import Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.models import model_from_json
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def find_patches_from_slide(slide_path, result_folder, filter_non_tissue):
    """Returns a dataframe of all patches in slide

    :param slide_path: the path for slide file
    :type slide_path: string
    :param result_folder: the place to save the results
    :type result_folder: string
    :param filter_non_tissue: if get rid of the no-tissue region
    :type filter_non_tissue: boolean

    :return: a dataframe with the columns:
        slide_path: path of slide
        is_tissue: sample contains tissue
        is_tumor: truth status of sample
        tile_loc: coordinates of samples in slide


    """
    print(slide_path)

    dimensions = []

    with openslide.open_slide(slide_path) as slide:
        dtotal = (slide.dimensions[0] / 224, slide.dimensions[1] / 224)
        thumbnail = slide.get_thumbnail((dtotal[0], dtotal[1]))
        thum = np.array(thumbnail)
        ddtotal = thum.shape
        dimensions.extend(ddtotal)
        hsv_image = cv2.cvtColor(thum, cv2.COLOR_RGB2HSV)
        # when the image was read into memory by opencv imread function, the array will switch columns between first and third columns. However, if the array already loded into memory.
        # and it is stored as rgb, it is ok to use RGB instead of BGR
        #hsv_image = cv2.cvtColor(thum, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        hthresh = threshold_otsu(h)
        sthresh = threshold_otsu(s)
        vthresh = threshold_otsu(v)
        # be min value for v can be changed later. according to dayong wang's paper, v value should be set to full range
        minhsv = np.array([hthresh, sthresh, 0], np.uint8)
        maxhsv = np.array([180, 255, 255], np.uint8)
        thresh = [minhsv, maxhsv]
        # print(thresh)
        # extraction the countor for tissue

        rgbbinary = cv2.inRange(hsv_image, thresh[0], thresh[1])
        _, contours, _ = cv2.findContours(
            rgbbinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxtcols = ['xmin', 'xmax', 'ymin', 'ymax']
        bboxt = pd.DataFrame(columns=bboxtcols)
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            bboxt = bboxt.append(
                pd.Series([x, x+w, y, y+h], index=bboxtcols), ignore_index=True)
            bboxt = pd.DataFrame(bboxt)

        xxmin = list(bboxt['xmin'].get_values())
        xxmax = list(bboxt['xmax'].get_values())
        yymin = list(bboxt['ymin'].get_values())
        yymax = list(bboxt['ymax'].get_values())

        xxxmin = np.min(xxmin)  # xxxmin = math.floor((np.min(xxmin))*224)
        xxxmax = np.max(xxmax)  # xxxmax = math.floor((np.max(xxmax))*224)
        yyymin = np.min(yymin)  # yyymin = math.floor((np.min(yymin))*224)
        yyymax = np.max(yymax)  # yyymax = math.floor((np.max(yymax))*224)

        dcoord = (xxxmin, xxxmax, yyymin, yyymax)
        # print(dcoord)
        dimensions.extend(dcoord)

        # bboxt = math.floor(np.min(xxmin)*256), math.floor(np.max(xxmax)*256), math.floor(np.min(yymin)*256), math.floor(np.max(yymax)*256)

        samplesnew = pd.DataFrame(rgbbinary)
        # samplesnew = pd.DataFrame(pd.DataFrame(
        # np.array(thumbnail.convert('L'))))
        # print(samplesnew)
        # very critical: y value is for row, x is for column
        samplesforpred = samplesnew.loc[yyymin:yyymax, xxxmin:xxxmax]
        # tissue_patches_bounding_box = tissue_patches.loc[yyymin:yyymax, xxxmin:xxxmax]

        # samplesforpred2 = samplesforpred*224
        dsample = samplesforpred.shape

        dimensions.extend(dsample)
        # print(dimensions)
        np.save('%s/%s' %
                (result_folder, osp.splitext(osp.basename(slide_paths[i]))[0]), dimensions)

        # print(samplesforpred)

        image_patch_index = pd.DataFrame(samplesforpred.stack())

        image_patch_index['is_tissue'] = image_patch_index[0]

        # print(image_patch_index)

        image_patch_index['tile_loc'] = list(image_patch_index.index)

        image_patch_index['slide_path'] = slide_path

        # tissue_patches_bounding_box_stack = pd.DataFrame(tissue_patches_bounding_box.stack())

        image_patch_index.reset_index(inplace=True, drop=True)

        # print(image_patch_index)

        if filter_non_tissue:

            image_patch_index = image_patch_index[image_patch_index.is_tissue != 0]

        image_patch_index.to_pickle(
            '%s/%s.pkl' % (result_folder, osp.splitext(osp.basename(slide_path))[0]))

    return image_patch_index


if __name__ == "__main__":

    slide_dir = '/raida/wjc/CAMELYON16/training/normal/'
    slide_paths = glob.glob(osp.join(slide_dir, '*.tif'))
    slide_paths.sort()
    # result_folder = "/Users/liw17/Documents/pred_dim_031tqdm(4/traig/normal/"
    result_folder = '/raidb/wli/testing_1219/patch_index/normal/'

    for i in tqdm(range(len(slide_paths))):

        find_patches_from_slide(
            slide_paths[i], result_folder, filter_non_tissue=False)
