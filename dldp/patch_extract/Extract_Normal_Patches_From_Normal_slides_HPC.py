#!/home/wli/env python3
# -*- coding: utf-8 -*-

"""
Title: Extract_Normal_Patches_From_Normal_Slides_HPC
====================================================
Created: 10-31-2019
Python-Version: 3.5

Description:
------------
This module is used to generate extract normal image patches from
normal WSI images on HPC (cluster) only. Each node will get one WSI
image done.

Request:
--------
This module requests the library module: patch_extractor.
This module also request a shell script to allocate the tasks.

"""
import openslide
# scipy.misc.imsave is deprecated! imsave is deprecated in SciPy 1.0.0,
# and will be removed in 1.2.0. Use imageio.imwrite instead.
# from scipy.misc import imsave as saveim
# before importing HDFStore, make sure 'tables' is installed by pip3 install tables
import os
import Patch_Extractor as PE

# before importing HDFStore, make sure 'tables' is installed by pip3 install tables


if __name__ == "__main__":

    # set the path for the normal slides
    slide_path_normal = '/home/wli/Downloads/CAMELYON16/training/normal'
    index_path = '/home/wli/pred_dim_10_26_18/normal'

    # set the folder to store the patches
    destination_folder_normal = '/home/wli/Downloads/normal_256'

    # put all the normal slides into a list
    normal_slide_paths = PE.slides_for_patch_extraction(
        slide_path_normal, 'tif')

    # patch size
    crop_size = [256, 256]

    # set the list of slides from which patches will be extracted
    slide_path_for_extraction = normal_slide_paths

    # correlate the slide with task_id of HPC
    taskid = int(os.environ['SGE_TASK_ID'])
    single_slide_for_patch_extraction = slide_path_for_extraction[taskid-1]

    # create a folder for each slide to store their patches
    des_folder_normal_patches = PE.create_folder(
        single_slide_for_patch_extraction, destination_folder_normal)

    # read wsi slide using openslide
    slide = openslide.open_slide(single_slide_for_patch_extraction)

    # get the bounding box for tissue region only
    bbox_tissue = PE.bbox_generation_tissue(slide)

    # get the threshold for tissue segmentation
    thresh = PE.tissue_patch_threshold(slide)

    # extract patches
    PE.extract_normal_patches_from_normal_slide(slide, thresh, crop_size, bbox_tissue, des_folder_normal_patches,
                                                single_slide_for_patch_extraction)
