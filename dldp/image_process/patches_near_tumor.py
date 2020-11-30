#!/home/wli/env python3
# -*- coding: utf-8 -*-
"""
Title: patches near tumor
=========================

Created: 10-31-2019
Python-Version: 3.5, 3.6

Description:
------------

This module is used to extract image patches near the border of tumor
region.

Inputs:
*******

    slide_path = '/raida/wjc/CAMELYON16/training/tumor'
    anno_path = '/raida/wjc/CAMELYON16/training/lesion_annotations'
    mask_dir = '/raida/wjc/CAMELYON16/training/masking'

Output:
*******

    patch_near_tumor_dir = '/raidb/wli/testing_1219/patch_near_tumor'

"""

import numpy as np
import pandas as pd
import os.path as osp
import openslide
# from scipy.misc import imsave as saveim
from imageio import imwrite as saveim
import glob
import cv2 as cv2
import xml.etree.ElementTree as et
from tqdm import tqdm
#########################################
import dldp.utils.fileman as fm


def convert_xml_df(file):
    """
    To convert xml file to a dataframe with all the coordinates

    :param file: xml file with the annotations from pathologists
    :type file: string
    :returns: all coordinates
    :rtype: dataframe

    """
    parseXML = et.parse(file)
    root = parseXML.getroot()
    dfcols = ['Name', 'Order', 'X', 'Y']
    df_xml = pd.DataFrame(columns=dfcols)
    for child in root.iter('Annotation'):
        for coordinate in child.iter('Coordinate'):
            Name = child.attrib.get('Name')
            Order = coordinate.attrib.get('Order')
            X_coord = float(coordinate.attrib.get('X'))
            Y_coord = float(coordinate.attrib.get('Y'))
            df_xml = df_xml.append(
                pd.Series([Name, Order, X_coord, Y_coord], index=dfcols), ignore_index=True)
            df_xml = pd.DataFrame(df_xml)
    return (df_xml)


def region_crop(slide, truth, crop_size, coord):
    """
    To extract all the four regions around a (x, y) coordinate together
    with the corresponding mask files.

    :param slide: the object created by openslide
    :type slide: object
    :param truth: the object created by openslide
    :type truth: object
    :param crop_size: the size of image patch to be extracted
    :type crop_size: list
    :param coord: a (x, y) coordinate
    :type coord: list
    :returns: four images patches and four mask images with the index
    :rtype: tuple

    """

    # width, height = slide.level_dimensions[0]
    dy, dx = crop_size
    x, y = coord
    x = int(x)
    y = int(y)
    # print(x, y)
    index = [[x, y], [x, y+dy-1], [x-dx+1, y], [x-dx+1, y+dy-1]]
    # print(index)
    #cropped_img = (image[x:(x+dx), y:(y+dy),:], rgb_binary[x:(x+dx), y:(y+dy)], mask[x:(x+dx), y:(y+dy)])
    rgb_image1 = slide.read_region((x, y), 0, crop_size)
    rgb_mask1 = truth.read_region((x, y), 0, crop_size)
    rgb_mask1 = (cv2.cvtColor(np.array(rgb_mask1),
                              cv2.COLOR_RGB2GRAY) > 0).astype(int)

    rgb_image2 = slide.read_region((x, y+dy-1), 0, crop_size)
    rgb_mask2 = truth.read_region((x, y+dy-1), 0, crop_size)
    rgb_mask2 = (cv2.cvtColor(np.array(rgb_mask2),
                              cv2.COLOR_RGB2GRAY) > 0).astype(int)

    rgb_image3 = slide.read_region((x-dx+1, y), 0, crop_size)
    rgb_mask3 = truth.read_region((x-dx+1, y), 0, crop_size)
    rgb_mask3 = (cv2.cvtColor(np.array(rgb_mask3),
                              cv2.COLOR_RGB2GRAY) > 0).astype(int)

    rgb_image4 = slide.read_region((x-dx+1, y+dy-1), 0, crop_size)
    rgb_mask4 = truth.read_region((x-dx+1, y+dy-1), 0, crop_size)
    rgb_mask4 = (cv2.cvtColor(np.array(rgb_mask4),
                              cv2.COLOR_RGB2GRAY) > 0).astype(int)

    # rgb_mask = (cv2.cvtColor(np.array(rgb_mask),
    #                         cv2.COLOR_RGB2GRAY) > 0).astype(int)
    # cropped_img = image[x:(x+dx), y:(y+dy),:]
    # cropped_binary = rgb_binary[x:(x+dx), y:(y+dy)]
    # cropped_mask = mask[x:(x+dx), y:(y+dy)]
    # print(index)
    return (rgb_image1, rgb_image2, rgb_image3, rgb_image4, rgb_mask1, rgb_mask2, rgb_mask3, rgb_mask4, index)


#sampletotal = pd.DataFrame([])

if __name__ == "__main__":

    slide_path = '/raida/wjc/CAMELYON16/training/tumor'
    # anno_path = '/Users/liw17/Documents/camelyon16/train/'
    anno_path = '/raida/wjc/CAMELYON16/training/lesion_annotations'
    mask_dir = '/raida/wjc/CAMELYON16/training/masking'
    slide_paths_total = glob.glob(osp.join(slide_path, '*.tif'))
    slide_paths_total.sort()

    crop_size = [224, 224]

    patch_near_tumor_dir = '/raidb/wli/testing_1219/patch_near_tumor'

    for i in tqdm(range(0, len(slide_paths_total))):
        # sampletotal = pd.DataFrame([])

        with openslide.open_slide(slide_paths_total[i]) as slide:

            new_folder = fm.creat_folder(osp.splitext(
                osp.basename(slide_paths_total[i]))[0], patch_near_tumor_dir)
            truth_slide_path = osp.join(mask_dir, osp.basename(
                slide_paths_total[i]).replace('.tif', '_mask.tif'))
            Anno_path_xml = osp.join(anno_path,
                                     osp.basename(slide_paths_total[i]).replace('.tif', '.xml'))

            with openslide.open_slide(str(truth_slide_path)) as truth:

                #slide = openslide.open_slide(slide_paths_total[i])
                annotations = convert_xml_df(str(Anno_path_xml))
                x_values = list(annotations['X'].get_values())
                y_values = list(annotations['Y'].get_values())
                coord = zip(x_values, y_values)
                coord = list(coord)

                m = 0
                while m in range(0, len(coord)):
                    r = region_crop(slide, truth, crop_size, coord[m])
                    for n in range(0, 4):
                        # selectlly save the image patches only outside of tumor regions
                        if (cv2.countNonZero(r[n+4]) == 0):

                            saveim('%s/%s_%d_%d.png' %
                                   (new_folder, osp.splitext(osp.basename(slide_paths_total[i]))[0], r[8][n][0], r[8][n][1]), r[n])

                            # io.imsave('/Users/liw17/Documents/new pred/%s_%d_%d_mask.png' % (osp.splitext(
                            #    osp.basename(slide_paths_total[i]))[0], r[8][n][0], r[8][n][1]), r[n+4])

                            # print(r[n])

                        m = m+5

                    # print(m)
