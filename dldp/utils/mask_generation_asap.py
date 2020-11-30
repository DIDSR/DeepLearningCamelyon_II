#!/home/wli/env python3
# -*- coding: utf-8 -*-
"""
Title: mask generation asap
===========================

Created: 10-31-2019
Python-Version: 3.5, 3.6

Description:
------------
This module is used to generate mask files (tiff).

Input:
******
 xml file (annotation)

Output:
*******
 mask files (pyramid, tiff)

Request:
--------
this module relys on:
         ASAP: https://github.com/computationalpathologygroup/ASAP
"""

import multiresolutionimageinterface as mir
import os.path as osp
import glob
import re

# please make sure the same number of files in the folder of tumor file and
# folder of annotation files
# please change the slide_path, anno_path, mask_path accordingly, and leave
# ever#ything else untouched.


# image_pair = zip(tumor_paths, anno_tumor_paths)
# image_pair = list(image_mask_pair)
class mask_gen_asap(object):
    """
    The class is used to generate training mask files from xml files.

    :param slide_path: the folder storing slides
    :type slide_path: str
    :param anno_path: the folder storing annotation files (xml)
    :type anno_path: str
    :param mask_path: the destination folder for the mask files
    :type mask_path: str
    :ivar slide_paths: all the paths of slides
    :vartype slide_paths: list
    :ivar anno_paths: all the paths of xml files
    :vartype anno_paths: list

    """
    reader = mir.MultiResolutionImageReader()

    def __init__(self, slide_path, anno_path, mask_path):
        """
        To initialize parameters

        :param slide_path: the folder storing slides
        :type slide_path: str
        :param anno_path: the folder storing annotation files (xml)
        :type anno_path: str
        :param mask_path: the destination folder for the mask files
        :type mask_path: str

        """

        self.slide_path = slide_path
        self.anno_path = anno_path
        self.mask_path = mask_path

        self.slide_paths = glob.glob(osp.join(self.slide_path, '*.tif'))
        self.slide_paths.sort()
        self.anno_paths = glob.glob(osp.join(anno_path, '*.xml'))
        self.anno_paths.sort()

    def mask_gen(self, slide_file, xml_file):
        """
        To generate mask file for one slide, and save the mask file.

        :param slide_file: the path of a WSI image
        :type slide_file: str
        :param xml_file: the path of a xml file for the annotation of WSI image
        :type xml_file: str
        :returns: the path of the mask file
        :rtype: str

        """

        mr_image = self.reader.open(slide_file)
        annotation_list = mir.AnnotationList()
        xml_repository = mir.XmlRepository(annotation_list)
        xml_repository.setSource(xml_file)
        xml_repository.load()
        annotation_mask = mir.AnnotationToMask()
        camelyon17_type_mask = False
        # Here 255 is used to generate mask file so that the tumor region is obvious.
        # if use '1' here, a binary maks file will be generated.
        label_map = {'metastases': 1, 'normal': 2} if camelyon17_type_mask else {
            '_0': 255, '_1': 255, '_2': 0}
        conversion_order = ['metastases', 'normal'] if camelyon17_type_mask else [
            '_0', '_1', '_2']
        output_path = osp.join(mask_path, osp.basename(
            slide_file).replace('.tif', '_mask.tif'))
        annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(
        ), mr_image.getSpacing(), label_map, conversion_order)

        return output_path

    def batch_gen(self):
        """
        To generate all the mask files for the slides in a folder

        :returns: None

        """

        for slide_file in self.slide_paths:
            xml_file = [x for x in self.anno_paths if re.search(osp.basename(
                slide_file).replace('.tif', '.xml'), x)][0]
            self.mask_gen(slide_file, xml_file)


if __name__ == "__main__":

    slide_path = '/raida/wjc/CAMELYON16/training/tumor'
    anno_path = '/raida/wjc/CAMELYON16/training/lesion_annotations'
    mask_path = '/raidb/wli/testing_1219/mask_for_training_wsi'

    make_mask = mask_gen_asap(slide_path, anno_path, mask_path)
    make_mask = make_mask.batch_gen()
