#!/home/wli/env python3
# -*- coding: utf-8 -*-

"""
Title: Extract_Normal_and_Tumor_Patches_From_Tumor_Slides
=========================================================
Created: 10-31-2019
Python-Version: 3.5, 3.6

Description:
------------
This module is used to extract normal and tumor image patches
from tumor WSI images.

Inputs:
*******

    slide_path_tumor = '/raida/wjc/CAMELYON16/training/tumor'
    mask_dir = '/raidb/wli/Final_Results/Display/train_masking'
    anno_dir = '/raida/wjc/CAMELYON16/training/lesion_annotations'

Output:
*******
    destination_folder_normal = '/raidb/wli/tumor_slide_normal_256_test'
    destination_folder_tumor = '/raidb/wli/tumor_slide_tumor_256_test'
    destination_folder_tumor_mask = '/raidb/wli/tumor_slide_tumor_256_mask_test'

Request:
--------
This module requests the library module: patch_extractor

Note:
-----
Mask image patches will be saved for quality control only. These patches
will not be used at all for GoogleNet training.

Mask image patches are generated as binary images. But when some pixels
are "0"s, and some pixels are "1"s, the "1"s will be saved as "255"s.

Some tumor slides can't have 1000 tumor slides to be extracted because
the tumor region is too small. The program needs to stop at these WSIs
and start mannually from next WSI.

"""
import os.path as osp
import openslide
# scipy.misc.imsave is deprecated! imsave is deprecated in SciPy 1.0.0,
# and will be removed in 1.2.0. Use imageio.imwrite instead.
# from scipy.misc import imsave as saveim
# before importing HDFStore, make sure 'tables' is installed by pip3 install tables
import Patch_Extractor as PE


if __name__ == "__main__":

    slide_path_tumor = '/raida/wjc/CAMELYON16/training/tumor'
    mask_dir = '/raidb/wli/Final_Results/Display/train_masking'
    anno_dir = '/raida/wjc/CAMELYON16/training/lesion_annotations'

    # The following destination_folders needs to be changed to the
    # the directories for the results
    destination_folder_normal = '/raidb/wli/testing_1219/normal_patches_from_tumor_slides'
    destination_folder_tumor = '/raidb/wli/testing_1219/tumor_patches_from_tumor_slide'
    destination_folder_tumor_mask = '/raidb/wli/testing_1219/tumor_patches_from_tumor_slides_mask'

    tumor_slide_paths = PE.slides_for_patch_extraction(slide_path_tumor, 'tif')
    # print(tumor_slide_paths)
    crop_size = [256, 256]
    slide_path_for_extraction = tumor_slide_paths
    # while loop is used here because 2-3 slides have very samll tumor region. It is
    # impossible to extract 1000 patches. The loop need to stop mannually and move
    # to next slide.
    i = 0
    while i < len(slide_path_for_extraction):

        single_slide_for_patch_extraction = slide_path_for_extraction[i]

        des_folder_normal_patches = PE.create_folder(
            single_slide_for_patch_extraction, destination_folder_normal)
        des_folder_tumor_patches = PE.create_folder(
            single_slide_for_patch_extraction, destination_folder_tumor)
        des_folder_tumor_patches_mask = PE.create_folder(
            single_slide_for_patch_extraction, destination_folder_tumor_mask)
        # sampletotal = pd.DataFrame([])single_slide_for_patch_extraction
        slide = openslide.open_slide(single_slide_for_patch_extraction)
        thresh = PE.tissue_patch_threshold(slide)
        bbox_tumor_region = PE.bbox_generation_tumor(
            single_slide_for_patch_extraction, anno_dir)

        bbox_tissue = PE.bbox_generation_tissue(slide)
        print(bbox_tissue)
        mask_path = osp.join(mask_dir, osp.basename(
            single_slide_for_patch_extraction).replace('.tif', '_mask.tif'))
        ground_truth = openslide.open_slide(str(mask_path))
        # normal patches will be extracted first
        PE.extract_normal_patches_from_tumor_slide(
            slide, ground_truth, crop_size, thresh, bbox_tissue, des_folder_normal_patches, single_slide_for_patch_extraction)
        PE.extract_tumor_patches_from_tumor_slide(slide, ground_truth, crop_size, thresh, bbox_tumor_region,
                                                  des_folder_tumor_patches, des_folder_tumor_patches_mask, single_slide_for_patch_extraction)
        i += 1
