import unitest
import pytest
import openslide
import os.path as osp

import dldp.patch_extract.Patch_Extractor as PE


def test_patch_extract():
    # the following code is for patch extraction from validation
    crop_size = [256, 256]
    slide_path_tumor = '/home/wli/Downloads/CAMELYON16/training/tumor/validation'
    slide_path_normal = '/home/wli/Downloads/CAMELYON16/training/normal/validation'
    normal_slide_paths = PE.slides_for_patch_extraction(slide_path_normal)
    tumor_slide_paths = PE.slides_for_patch_extraction(slide_path_tumor)
    slide_path_for_extraction = normal_slide_paths + tumor_slide_paths

    i = 0
    while i < len(slide_path_for_extraction):

        single_slide_for_patch_extraction = slide_path_for_extraction[i]
        slide = openslide.open_slide(single_slide_for_patch_extraction)
        slide_contains_tumor = osp.basename(
            single_slide_for_patch_extraction).startswith('tumor_')

        bbox_tissue = PE.bbox_generation_tissue(slide)
        thresh = PE.tissue_patch_threshold(slide)

        if slide_contains_tumor:

            folder_tumor_patches = PE.create_folder(
                single_slide_for_patch_extraction, destination_folder_tumor)
            folder_normal_patches = PE.create_folder(
                single_slide_for_patch_extraction, destination_folder_normal)
            folder_tumor_patches_mask = PE.create_folder(
                single_slide_for_patch_extraction, destination_folder_mask)

            bbox_tumor = PE.bbox_generation_tumor(
                single_slide_for_patch_extraction, anno_dir)

            mask_path = osp.join(mask_dir, osp.basename(
                single_slide_for_patch_extraction).replace('.tif', '_mask.tif'))
            ground_truth = openslide.open_slide(str(mask_path))

            PE.extract_tumor_patches_from_tumor_slide(slide, thresh, bbox_tumor, folder_tumor_patches,
                                                      folder_tumor_patches_mask, single_slide_for_patch_extraction)
            PE.extract_normal_patches_from_tumor_slide(slide, thresh, bbox_tissue, folder_normal_patches,
                                                       single_slide_for_patch_extraction)

        else:
            folder_normal_patches = PE.create_folder(
                single_slide_for_patch_extraction, destination_folder_normal)
            slide = openslide.open_slide(slide_paths_total[i])
            # c=[]
            PE.extract_normal_patches_from_normal_slide(slide, thresh, bbox_tissue, folder_normal_patches,
                                                        single_slide_for_patch_extraction)

        i = i + 1


if __name__ == "__main__":

    test_patch_extract()
