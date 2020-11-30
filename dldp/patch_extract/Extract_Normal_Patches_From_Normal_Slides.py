#!/home/wli/env python3
# -*- coding: utf-8 -*-

"""
Title: Extract_Normal_Patches_From_Normal_Slides
================================================
Created: 10-31-2019
Python-Version: 3.5
Description:
------------
This module is used to extract normal image patches from
normal WSI images.

Input:
******
    slide_path_normal = '/raida/wjc/CAMELYON16/training/normal'

Output:
*******
    destination_folder_normal = '/raidb/wli/testing_1219/normal_256_pathces'

Request:
--------
This module requests the library module: patch_extractor

"""
import openslide
# scipy.misc.imsave is deprecated! imsave is deprecated in SciPy 1.0.0,
# and will be removed in 1.2.0. Use imageio.imwrite instead.
# from scipy.misc import imsave as saveim
# before importing HDFStore, make sure 'tables' is installed by pip3 install tables
import Patch_Extractor as PE

if __name__ == "__main__":

    slide_path_normal = '/raida/wjc/CAMELYON16/training/normal'
    destination_folder_normal = '/raidb/wli/testing_1219/normal_patches_from_normal_slides'

    normal_slide_paths = PE.slides_for_patch_extraction(
        slide_path_normal, 'tif')
    crop_size = [256, 256]
    slide_path_for_extraction = normal_slide_paths
    i = 0
    while i < len(slide_path_for_extraction):

        single_slide_for_patch_extraction = slide_path_for_extraction[i]
        des_folder_normal_patches = PE.create_folder(
            single_slide_for_patch_extraction, destination_folder_normal)
        # sampletotal = pd.DataFrame([])single_slide_for_patch_extraction
        slide = openslide.open_slide(single_slide_for_patch_extraction)
        bbox_tissue = PE.bbox_generation_tissue(slide)
        thresh = PE.tissue_patch_threshold(slide)

        PE.extract_normal_patches_from_normal_slide(
            slide, thresh, crop_size, bbox_tissue, des_folder_normal_patches, single_slide_for_patch_extraction)

        i = i+1
