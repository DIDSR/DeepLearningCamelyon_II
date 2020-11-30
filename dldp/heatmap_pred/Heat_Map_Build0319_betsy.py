# !/home/wli/env python3
# -*- coding: utf-8 -*-
"""
Heat map stiching
=================

Created: 10-31-2019
Python-Version: 3.5, 3.6

Description:
------------

After prediction, each patch from a WSI image has one prediction matrix (for example, 14x14). This script is
used to put all these small matrix into a big map corresponding to a rectangle tissue region of a
WSI image.

Inputs:
*****

dimensions: the locations of dimension files
Folder_Prediction_Results: the location of the prediction for individual patches
slide_category: the category of the slide, for example, 'tumor', 'normal', 'test'
Stride: the skipped pixels when prediction, for example, 16, 64 

Output:
*******

Folder_Heatmap: the folder to store the stitched heatmap

Note
----
The following files for "dimensions" are necessary to perform the task:
'.../pred_dim_0314/training-updated/normal/dimensions',
'.../pred_dim_0314/training-updated/tumor/dimensions',
'.../pred_dim_0314/testing'

These files store the dimension of the heatmap and location of the heatmap in the WSI image.

These flies were generated by utils/patch_index.py

The dimension files need to be generated separately for "normal", "tumor",
and "test" WSIs.


"""
import os
import os.path as osp
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
# import multiprocess as mlp
import re

# strike = 16
# dimension_of_pred_patch = 224/strike = 14


def stitch_preded_patches(dimension_files, pred_folder, Folder_Heatmap, Stride):
    """
    stitching the prediction based on each small patches to a big heatmap

    :param dimension_files: a list of all the dimension files for one category of slides, foe example, 'tumor' 
    :type dimension_files: list
    :param pred_folder: the folder having all the patch prediction results for a single WSI image.
    :type pred_folder: str
    :param Folder_Heatmap: the folder to store the big stitched heatmap.
    :type Folder_Heatmap: str
    :param stride: the stride during prediction
    :type stride: int

    :return: no return

    :note: two files will saved to the Folder_Heatmap:
            1. the stitched heatmap in npy format
            2. the heatmap picture in png format
    """

    # find the dimension file according to the name of folder that store all the predicted npy file

    dimension_paths = [x for x in dimension_files if re.search(
        osp.basename(pred_folder), x)]
    print(dimension_paths)
    dimension_path = dimension_paths[0]

    dimension = np.load(dimension_path)

    # create a big heat map for small matrix to be filled in

    num_of_pred_per_patch = int(224/Stride)

    heat_map_big = np.zeros([dimension[7]*num_of_pred_per_patch,
                             dimension[8]*num_of_pred_per_patch], dtype=np.float32)

    # generate a list of all npy files inside one folder.

    files = glob.glob(osp.join(pred_folder, '*.npy'))
    files.sort()

    # create a empty list to store all the npy files. each npy file is a list of (160, 14, 14) list.
    heat_map = []

    for file in files:
        regions = np.load(file)
        heat_map.extend(regions)
    # now we start to do reshaping

    heat_map_array = np.array(heat_map)
    # These are critical steps to construct heatmap in a time saving manner.
    heat_map_reshape = heat_map_array.reshape(
        dimension[7], dimension[8], num_of_pred_per_patch, num_of_pred_per_patch)

    b = heat_map_reshape.transpose((0, 2, 1, 3))

    c = b.reshape(heat_map_big.shape[0], heat_map_big.shape[1])

    np.save('%s/%s' % (Folder_Heatmap, osp.basename(pred_folder)), c)
    #np.save('/scratch/weizhe.li/heat_map/Method_II_Model_II_norm/test_0516/%s' % (osp.basename(pred_folder)), c)

    matplotlib.image.imsave('%s/%s.png' %
                            (Folder_Heatmap, osp.basename(pred_folder)), c)
    #matplotlib.image.imsave('/scratch/weizhe.li/heat_map/Method_II_Model_II_norm/test_0516/%s.png' % (osp.basename(pred_folder)), c)


if __name__ == "__main__":

    taskid = int(os.environ['SGE_TASK_ID'])
    Folder_Prediction_Results = '/scratch/weizhe.li/Pred_Storage/baseline/normal_baseline'
    dirs = os.listdir(Folder_Prediction_Results)
    #dirs = os.listdir('/scratch/weizhe.li/Pred_Storage/Full_Resolution_Pred/test_0516_II_real_II')

    dirs.sort()
    slide_categories = ['normal', 'tumor', 'test']

    dimensions = {'normal': '/home/weizhe.li/li-code4hpc/pred_dim_0314/training-updated/normal/dimensions',
                  'tumor': '/home/weizhe.li/li-code4hpc/pred_dim_0314/training-updated/tumor/dimensions',
                  'test': '/home/weizhe.li/li-code4hpc/pred_dim_0314/testing/dimensions'
                  }
    # the dimension of the rectangle tissue region from WSI images was stored individually as npy file in the folder of pred_dim_0314
    slide_category = slide_categories[0]
    Folder_dimension = dimensions[slide_category]

    dimension_files = glob.glob(osp.join(Folder_dimension, '*.npy'))
    dimension_files.sort()
    print(dimension_files)

    # Here is the folder for prediction results.The prediction results are organized into folders. Each folder corresponds to a WSI image.
    # Inside folder, there are large number of npy files. each file
    # is a 14x14x160 array.

#    Pred_folders = '/scratch/weizhe.li/Pred/tumor_0324'
    #Folder_Heatmap = '/scratch/weizhe.li/heat_map/Color_Norm_Methods/Reinhard/test_0927'
    Folder_Heatmap = '/scratch/weizhe.li/heat_map/Color_Normal_Assess/baseline/normal'
    Stride = 56

    i = taskid - 1

    if i < len(dirs):
        pred_folder = osp.join(Folder_Prediction_Results, dirs[i])

        print(pred_folder)
        stitch_preded_patches(dimension_files, pred_folder,
                              Folder_Heatmap, Stride)