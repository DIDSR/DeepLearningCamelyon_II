#!/home/wli/env python3
# -*- coding: utf-8 -*-

"""
Title: convert XML to binary mask
=================================
Created: 10-31-2019
Python-Version: 3.5, 3.6

Description:
------------
This module is used to generate binary mask files from the annotations (XML files).
This module is needed for generating mask files for testing WSIs because the code coming with ASAP showed
errors for mask generation on certain testing WSIs.

Inputs:
*******

    xml_folder = '/raida/wjc/CAMELYON16/testing/lesion_annotations'
    slide_folder = '/raida/wjc/CAMELYON16/testing/images'
    level = 5 : the level of zoom (0 is the highest; 5 is the 32x
                                   downsampled image from level 0).

Output:
*******
    result_folder = '/raidb/wli/testing_1219/mask_for_testing_wsi'


Note:
-----
If you need more information about how ElementTree package handle XML file,
please follow the link:
#https://docs.python.org/3/library/xml.etree.elementtree.html

"""


import math
import glob
import pandas as pd
import xml.etree.ElementTree as et
from pandas import DataFrame
import openslide
import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging
import os.path as osp
import sys
sys.path.append('/home/weizhe.li/dldp/utils/logman')
sys.path.append('/home/weizhe.li/dldp/utils')
# setup_logging

import fileman as fm
import logger_management
from logger_management import log_with_template
from logger_management import StreamToLogger
from logger_management import setup_logging

# import multiresolutionimageinterface as mir


# reader = mir.MultiResolutionImageReader()
# mr_image = reader.open('/home/wli/Downloads/tumor_036.tif')
# Ximageorg, Yimageorg = mr_image.getDimensions()
# dims = mr_image.getLevelDimensions(4)
# Ximage = (Ximage+240//2)//240
# Ximage = 4000
# Yimage = (Yimage+240//2)//240
# Yimage = 2000


class mask_generator(object):
    """
    The class is used to generate a single mask file (not pyramid) based
    on xml file.

    """

    def __init__(self, xml_file, level, dims):
        """
        variables initialization

        :param xml_file:
        :param level:
        :param dims:
        :param result_folder:

        """

        self.xml_file = xml_file
        self.level = level
        self.dims = dims
        self.result_folder = result_folder

    def convert_xml_df(self):
        """
        To convert a xml file to a series of dataframes in a tuple.

        :return: df_xml: x, y coordinates
        :rtype: dataframe
        """
        down_sample = 2**self.level
        parseXML = et.parse(self.xml_file)
        root = parseXML.getroot()
        dfcols = ['Name', 'Order', 'X', 'Y']
        df_xml = pd.DataFrame(columns=dfcols)
        for child in root.iter('Annotation'):
            for coordinate in child.iter('Coordinate'):
                Name = child.attrib.get('Name')
                Order = coordinate.attrib.get('Order')
                X_coord = float(coordinate.attrib.get('X'))

                X_coord = X_coord//down_sample
                
                #X_coord = X_coord/down_sample

                Y_coord = float(coordinate.attrib.get('Y'))

                Y_coord = Y_coord//down_sample
                #Y_coord = Y_coord/down_sample

                df_xml = df_xml.append(pd.Series(
                    [Name, Order, X_coord, Y_coord], index=dfcols),
                    ignore_index=True)  # type: DataFrame
                df_xml = pd.DataFrame(df_xml)
        print('redundent xml:', df_xml.shape)
        return df_xml

    # x_values = list(annotations['X'].get_values())
    # y_values = list(annotations['Y'].get_values())
    # xy = list(zip(x_values,y_values))

    def points_collection(self, annotations):
        """
        remove the duplicated coordinates due to the down_sampling
        :param duplicate:
        :return: list with no duplicates
        :rtype: list
        """
        final_name_list = list(annotations['Name'].unique())
        
        coxy = [[] for x in range(len(final_name_list))]
        for index, n in enumerate(final_name_list):
            newx = annotations[annotations['Name'] == n]['X']
            newy = annotations[annotations['Name'] == n]['Y']
            newxy = list(zip(newx, newy))
            coxy[index] = np.array(newxy, dtype=np.int32)

        return (coxy, final_name_list)

    def mask_gen(self, coxy, result_folder):
        """
        generate a binary mask file

        :param final_list: the down-sampled annotation
        :type final_list: list
        :param result_folder:
        :type result_folder:str
        :return: mask file
        :rtype: tif file
        """


        # image = cv2.imread('/home/wli/Downloads/tumor_036.xml', -1)
        canvas = np.zeros((int(self.dims[1]//2**self.level), int(self.dims[0]//2**self.level)), np.uint8)
        # canvas = np.zeros((int(dims[1]/32), int(dims[0]/32)), np.uint8)

        # tile =mr_image.getUCharPatch(0, 0, dims[0], dims[1], 4)
        # canvas = np.zeros((Ximage, Yimage, 3), np.uint8) # fix the division
        # coords = np.array([xy], dtype=np.int32)

        # cv2.drawContours(canvas, [coords],-1, (0,255,0), -1)

        # cv2.drawContours(canvas, coxy, -1, (255, 255, 255), 10)
        # cv2.drawContours(canvas, coxy, -1, (255, 255, 255), CV_FILLED)
        cv2.fillPoly(canvas, pts=coxy, color=(255, 255, 255))
        # cv2.polylines(canvas, coxy, isClosed=True, color=(255,255,255),
        # thickness=5)

        cv2.imwrite('%s/%s.png' % (result_folder,
                                   osp.splitext(osp.basename(self.xml_file))[0]), canvas)
# cv2.imshow("tile", tile);cv2.waitKey();cv2.destroyAllWindows()

# cv2.fillConvexPoly(mask, coords,1)
# mask = mask.astype(np.bool)

# output = np.zeros_like(image)
# output[mask] = image[mask]

# cv2.imshow('image',output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
class lesion_count(mask_generator):
    def __init__(self):
        super().__init__
    def gen_lesion_table(self, coxy, final_name_list):
        lesion_total = []
        for coord, lesion_name in list(zip(coxy, contour_names)):
    
            M = cv2.moments(coord)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            lesion_total.append([lesion_name, (cX, cY), cv2.contourArea(coord)])
        return lesion_total
        


if __name__ == '__main__':

    # setup_logging
    module_name = sys.modules['__main__'].__file__
    log_template_path = '/home/weizhe.li/dldp/utils/logman/logging_config.yaml'
    # The log template is also inculded in this package.
    # log_template_path = '.../dldp/utils/logman/logging_config.yaml'
    logger = log_with_template(log_template_path, module_name)
    # xml_folder  = '/home/wzli/Downloads/CAMELYON16/testing/
    # lesion_annotations/'
    # xml_folder  = '/home/wzli/Downloads/CAMELYON16/training/
    # lesion_annotations_training/'
    xml_folder = '/projects01/wxc4/wli/CAMELYON16/lesion_annotations'
    xml_paths = fm.list_file_in_dir_II(xml_folder, 'xml')
    logger.debug('the fist xml file is %s' % xml_paths[0])
    # xml_paths = glob.glob(osp.join(xml_folder, '*.xml'))
    # xml_paths.sort()
    # slide_folder = '/home/wzli/Downloads/CAMELYON16/testing/images/'
    slide_folder = '/projects01/wxc4/CAMELYON16-training/tumor'
    result_folder = '/projects01/wxc4/wli/CAMELYON16/lesion_counts'

    created_folder = fm.creat_folder('', result_folder)

    # slide_paths = glob.glob(osp.join(slide_folder, '*.tif'))
    level = 5
    ############################lesion count#######################################
    lesion_total = []
    col_names = ['slide_name', 'lesion_name', 'centroid', 'area']
    for xml_file in xml_paths:
        slide_name = osp.basename(xml_file.replace('.xml', '.tif'))
        slide_path = osp.join(slide_folder, slide_name)
        wsi_image = openslide.open_slide(slide_path)
        dims = wsi_image.dimensions

        mask_gen = mask_generator(xml_file, level, dims)
        annotations = mask_gen.convert_xml_df()
        final_annotations, _ = mask_gen.points_collection(annotations)
        
        # mask_gen.mask_gen(final_annotations, reult_folder)
        lesion_stat = lesion_count(xml_file, level, dims)
        annotations = lesion_stat.convert_xml_df()
        final_annotations, lesion_names = lesion_stat.points_collection(annotations)

        slide_lesions = lesion_stat.gen_lesion_table(final_annotations, lesion_names)

        lesion_total.append(slide_lesions)

        df_lesion_stat = pd.DataFrame(lesion_total, columns=col_names)
        df_lesion_stat.to_csv(result_folder)
