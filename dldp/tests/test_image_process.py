import pytest
import unittest
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

import dldp.image_process.color_normalization as cn
import dldp.utils.fileman as fm
import dldp.image_process.hard_negative_mining as hm

slide_dir = '/home/wli/Downloads/CAMELYON16/training/tumor'
grouth_truth_dir = '/home/wli/Downloads/CAMELYON16/masking'
mask_dir = '/home/wli/Downloads/CAMELYON16/masking'
index_path = '/home/wli/Documents/pred_dim_0314/training-updated/tumor/patch_index'

patch_index_paths = fm.list_file_in_dir_II(index_path, 'pkl')
slide_paths = fm.list_file_in_dir_II(slide_dir, 'tif')
ground_truth_paths = fm.list_file_in_dir_II(grouth_truth_dir, 'tif')

result_folder = '/home/weizheli/Documents/hnm'
model = load_model(
    '/home/wli/Training/Redo/0930/no_color_noise_0/googlenetv1_Vahadane_Total_Patch_Retrain_09.30.19 09:45_Origin-03-0.94.hdf5')
template_image_path = '/raidb/wli/tumor_st.png'
color_norm_method = "vahadane"
log_path = '/home/wli/log_files'


def test_hard_negative_mining():
    """
    test function for hard negative mining module

    """
    for patch_index_path in patch_index_paths:

        assert len(patch_index_paths) == 111
        assert type(patch_index_path) == 'string'

        all_samples, n_samples, slide, new_slide_path, ground_truth = hm.slide_level_info(
            patch_index_path, slide_paths, ground_truth_paths)

        assert len(all_samples.columns) == 6

        for index, all_samples_entry in all_samples.iterrows():

            xylarge = hm.coordinates(all_samples_entry, hm.pred_size)

            if all_samples_entry.is_tissue == 0:

                pred = 0
                truth = 0

            else:

                img = hm.generate_image_patches(slide, xylarge)
                truth = hm.generate_image_patches(ground_truth, xylarge)

                if color_norm_method:
                    fit = cn.color_normalization(
                        template_image_path, color_norm_method)

                    img = hm.color_norm(img, fit)

                else:
                    img = img

                img = hm.dim_exp(img)

                assert len(img.shape) == 4
                pred = model.predict(img)[:, 1]

                if np.count_no_zeros(truth):

                    truth = 1

                else:

                    truth = 0
            # update the dataframe with the new values
            all_samples.at[index, 'pred'] = pred
            all_samples.at[index, 'truth'] = truth
