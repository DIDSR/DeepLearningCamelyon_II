#!/home/wli/env python3
# -*- coding: utf-8 -*-
"""
Title: patch extractor
======================

Created: 10-31-2019
Python-Version: 3.5, 3.6

Description:
------------

This libray module provides functions for patch extraction.

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os.path as osp
import openslide
from pathlib import Path
# scipy.misc.imsave is deprecated! imsave is deprecated in SciPy 1.0.0,
# and will be removed in 1.2.0. Use imageio.imwrite instead.
#from scipy.misc import imsave as saveim
from imageio import imwrite as saveim
from skimage.filters import threshold_otsu
import glob
# before importing HDFStore, make sure 'tables' is installed by pip3 install tables
from pandas import HDFStore
from openslide.deepzoom import DeepZoomGenerator
import cv2 as cv2
from skimage import io as skio
import xml.etree.ElementTree as et
import math
import os


# go through all the file
def slides_for_patch_extraction(slide_folder, file_ext):
    """
    This function is to generate a list for all the slides in a folder.

    :param slide_folder: a folder storing WSI images.
    :type slide_folder: string
    :param file_ext: file type, for exaple, "tif"
    :type file_ext: string

    :return: slide_paths
    :rtype: a list including all the obsolute paths of the slides from a
            folder.

    """
    slide_paths = glob.glob(osp.join(slide_folder, '*.%s' % file_ext))
    slide_paths.sort()
    return slide_paths


def hsv_thumbnail(slide):
    """
    generate a HSV thumbnail image for WSI image with downsample of 32.
    The ratio of length and width of the image is still the same as the
    level 0 image.

    :param slide: the initialized slide oject from openslide
    :type slide: object

    :return: hsv image
    :rtype: array

    """
    thumbnail = slide.get_thumbnail(
        (slide.dimensions[0] / 32, slide.dimensions[1] / 32))
    thum = np.array(thumbnail)
    hsv_image = cv2.cvtColor(thum, cv2.COLOR_RGB2HSV)
    return hsv_image


def tissue_patch_threshold(slide):
    """
    get a threshold for tissue region

    :param slide: the initialized slide oject from openslide
    :type slide: objec
    :returns: threshold
    :rtype: list

    """
    hsv_image = hsv_thumbnail(slide)
    h, s, v = cv2.split(hsv_image)
    hthresh = threshold_otsu(h)
    sthresh = threshold_otsu(s)
    vthresh = threshold_otsu(v)
    # be min value for v can be changed later
    minhsv = np.array([hthresh, sthresh, 70], np.uint8)
    maxhsv = np.array([180, 255, vthresh], np.uint8)
    thresh = [minhsv, maxhsv]
    return thresh


def bbox_generation_tissue(slide):
    """
    generate a bounding box for tissue region in a WSI image

    :param slide: the initialized slide oject from openslide
    :type slide: object
    :returns: bbox_tissue, the coordinates for the four corners of
              the tissue region.
    :rtype: tuple

    """
    hsv_image = hsv_thumbnail(slide)
    # h, s, v = cv2.split(hsv_image)
    # hthresh = threshold_otsu(h)
    # sthresh = threshold_otsu(s)
    # vthresh = threshold_otsu(v)
    # be min value for v can be changed later
    # minhsv = np.array([hthresh, sthresh, 70], np.uint8)
    # maxhsv = np.array([180, 255, vthresh], np.uint8)
    # thresh = [minhsv, maxhsv]
    thresh = tissue_patch_threshold(slide)

    print(thresh)
    # extraction the countor for tissue

    rgbbinary = cv2.inRange(hsv_image, thresh[0], thresh[1])
    print(rgbbinary.shape)

    # old version of cv2.findContours gives three returns
    _, contours, _ = cv2.findContours(
        rgbbinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxtcols = ['xmin', 'xmax', 'ymin', 'ymax']
    bboxt = pd.DataFrame(columns=bboxtcols)
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        bboxt = bboxt.append(
            pd.Series([x, x + w, y, y + h], index=bboxtcols),
            ignore_index=True)
        bboxt = pd.DataFrame(bboxt)

    xxmin = list(bboxt['xmin'].get_values())
    xxmax = list(bboxt['xmax'].get_values())
    yymin = list(bboxt['ymin'].get_values())
    yymax = list(bboxt['ymax'].get_values())
    bbox_tissue = (math.floor(np.min(xxmin) * 32), math.floor(
        np.max(xxmax) * 32), math.floor(
        np.min(yymin) * 32), math.floor(np.max(yymax) * 32))
    print(str(bbox_tissue))
    return bbox_tissue


def bbox_generation_tumor(single_slide_for_patch_extraction, anno_dir):
    """
    generate a bounding box for tumor region. If several regions exist,
    a big bounding box will be generated to include all the regions.

    :param single_slide_for_patch_extraction: a slide for path extraction, a path
    :type single_slide_for_patch_extraction: string
    :param anno_dir: annotations files
    :type anno_dir: list
    :returns: bbox_tumor, the coordinates for the four corners of
              the tumor region.
    :rtype: tuple

    """
    Anno_pathxml = osp.join(anno_dir, osp.basename(
        single_slide_for_patch_extraction).replace('.tif', '.xml'))

    # slide = openslide.open_slide(single_slide_for_patch_extraction)
    annotations = convert_xml_df(str(Anno_pathxml))
    x_values = list(annotations['X'].get_values())
    y_values = list(annotations['Y'].get_values())
    bbox_tumor = (math.floor(np.min(x_values)), math.floor(np.max(x_values)),
                  math.floor(
        np.min(y_values)), math.floor(
        np.max(y_values)))
    return bbox_tumor


def convert_xml_df(file):
    """
    convert the xml file to a list of coordinates

    :param file: path for an xml file
    :returns: coordinates
    :rtype: tuple including all the coordinates

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
            # X_coord = X_coord - 30000
            # X_coord = ((X_coord)*dims[0])/Ximageorg
            Y_coord = float(coordinate.attrib.get('Y'))
            # Y_coord = Y_coord - 155000
            # Y_coord = ((Y_coord)*dims[1])/Yimageorg
            df_xml = df_xml.append(
                pd.Series([Name, Order, X_coord, Y_coord], index=dfcols), ignore_index=True)
            df_xml = pd.DataFrame(df_xml)
    return (df_xml)


def random_crop_tumor(slide, truth, thresh, crop_size, bbox):
    """
    The major function to extract image patches from tumor WSI images together
    with ground truth. This function is used for normal or tumor patch 
    extraction with its ground truth.

    :param slide: slide object created by openslide
    :type slide: object
    :param truth: ground truth object created by openslide
    :type param: object
    :param thresh: threshold for tissue region
    :type thresh: list
    :param crop_size: the size of image patch to be generated
    :type crop_size: list
    :param bbox: the coordinates of a bounding box
    :type bbox: tuple
    :returns: rgb_image, rgb_binary, rgb_mask, index
    :rtype: tuple

    :note: The "bbox" will the bbox for tissue region if extract normal
           patches; The "bbox" will be the bbox for tumor region if
           extract tumor patches.
    """
    # width, height = slide.level_dimensions[0]
    dy, dx = crop_size
    x = np.random.randint(bbox[0], bbox[1] - dx + 1)
    y = np.random.randint(bbox[2], bbox[3] - dy + 1)
    # x = np.random.choice(range(width - dx + 1), replace = False)
    # y = np.random.choice(range(height - dy +1), replace = False)
    index = [x, y]
    # print(index)
    # cropped_img = (image[x:(x+dx), y:(y+dy),:], rgb_binary[x:(x+dx), y:(y+dy)], mask[x:(x+dx), y:(y+dy)])
    rgb_image = slide.read_region((x, y), 0, crop_size)
    rgb_mask = truth.read_region((x, y), 0, crop_size)
    rgb_mask = (cv2.cvtColor(np.array(rgb_mask),
                             cv2.COLOR_RGB2GRAY) > 0).astype(int)
    rgb_array = np.array(rgb_image)
    hsv_rgbimage = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    rgb_binary = cv2.inRange(hsv_rgbimage, thresh[0], thresh[1])
    # cropped_img = image[x:(x+dx), y:(y+dy),:]
    # cropped_binary = rgb_binary[x:(x+dx), y:(y+dy)]
    # cropped_mask = mask[x:(x+dx), y:(y+dy)]
    # print(index)
    return (rgb_image, rgb_binary, rgb_mask, index)

# random_crop2 is not nessary.


def random_crop2(slide, truth, thresh, crop_size, bboxt):
    """
    The major function to extract image patches from WSI images together with
    ground truth. This function is used for normal patch extraction with its
    ground truth.

    :param slide: object generated using openslide
    :type slide: object
    :param truth: object generated using openslide
    :type truth: object
    :param thresh: the threshhold for tissue region
    :type thresh: list
    :param crop_size: the size of image patches to be extracted
    :type crop_size: list
    :param bboxt: the bounding box for tissue region
    :type bboxt: tuple
    :returns: rgb_image, rgb_binary, rgb_mask, index
    :rtype: tuple

    """
    # width, height = slide.level_dimensions[0]
    dy, dx = crop_size
    # print(bboxt[0], bboxt[1])
    x = np.random.randint(bboxt[0], bboxt[1] - dx + 1)
    y = np.random.randint(bboxt[2], bboxt[3] - dy + 1)
    # x = np.random.choice(range(width - dx + 1), replace = False)
    # y = np.random.choice(range(height - dy +1), replace = False)
    index = [x, y]
    # print(index)
    # cropped_img = (image[x:(x+dx), y:(y+dy),:], rgb_binary[x:(x+dx), y:(y+dy)], mask[x:(x+dx), y:(y+dy)])
    rgb_image = slide.read_region((x, y), 0, crop_size)
    rgb_mask = truth.read_region((x, y), 0, crop_size)
    rgb_mask = (cv2.cvtColor(np.array(rgb_mask),
                             cv2.COLOR_RGB2GRAY) > 0).astype(int)
    rgb_array = np.array(rgb_image)
    hsv_rgbimage = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    rgb_binary = cv2.inRange(hsv_rgbimage, thresh[0], thresh[1])
    # cropped_img = image[x:(x+dx), y:(y+dy),:]
    # cropped_binary = rgb_binary[x:(x+dx), y:(y+dy)]
    # cropped_mask = mask[x:(x+dx), y:(y+dy)]
    print(index)
    return (rgb_image, rgb_binary, rgb_mask, index)


def random_crop_normal(slide, thresh, crop_size, bbox_tissue):
    """
    The major function for image patch generation. This function is used to get
    image patches from normal WSI slides.

    :param slide: object generated by openslide
    :type slide: object
    :param thresh: the threshold for tissue region
    :type thresh: list
    :param crop_size: the size of image patches to be extracted
    :type crop_size: list
    :param bbox_tissue: the bounding box for tissue region
    :type bbox_tissue: tuple
    :returns: rgb_image, rgb_binary, index
    :rtype: tuple

    """
    # width, height = slide.level_dimensions[0]
    dy, dx = crop_size
    x = np.random.randint(bbox_tissue[0], bbox_tissue[1] - dx + 1)
    y = np.random.randint(bbox_tissue[2], bbox_tissue[3] - dy + 1)
    index = [x, y]
    # cropped_img = (image[x:(x+dx), y:(y+dy),:], rgb_binary[x:(x+dx), y:(y+dy)], mask[x:(x+dx), y:(y+dy)])
    rgb_image = slide.read_region((x, y), 0, crop_size)
    # rgb_mask = truth.read_region((x, y), 0, crop_size)
    # rgb_mask = (cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY) > 0).astype(int)
    # rgb_grey = np.array(rgb_image.convert('L'))
    # rgb_binary = (rgb_grey < thresh).astype(int)
    rgb_array = np.array(rgb_image)
    hsv_rgbimage = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    rgb_binary = cv2.inRange(hsv_rgbimage, thresh[0], thresh[1])
    # cropped_img = image[x:(x+dx), y:(y+dy),:]
    # cropped_binary = rgb_binary[x:(x+dx), y:(y+dy)]
    # cropped_mask = mask[x:(x+dx), y:(y+dy)]
    return (rgb_image, rgb_binary, index)


def testduplicates(list):
    """
    get rid of duplicate entries

    :param list: the list for duplication checking
    :type list: list
    :returns: the list with no duplication
    :rtype: list

    """
    for each in list:
        count = list.count(each)
        if count > 1:
            z = 0
        else:

            z = 1
    return z


def create_folder(single_slide_for_patch_extraction, destination_folder):
    """
    The function is used to create folder and store image patches. All the
    image patches extracted from the same slide will be saved in the same
    folder.

    :param single_slide_for_patch_extraction: the slide used to get image
                                              patches
    :type single_slide_for_patch_extraction: str
    :param destination_folder: the place to store all the extracted image
                               patches
    :return: the folder to be created
    :rtype: str

    """
    print(single_slide_for_patch_extraction)
    print(osp.splitext(osp.basename(single_slide_for_patch_extraction)))
    folder_to_create = osp.splitext(
        osp.basename(single_slide_for_patch_extraction))[0]
    print(folder_to_create)
    path_for_folder = osp.join(destination_folder, folder_to_create)
    print(path_for_folder)
    try:
        os.makedirs(path_for_folder)
    except Exception:
        print('folder exist, skipped')

    return path_for_folder


# sampletotal = pd.DataFrame([])
def extract_normal_patches_from_normal_slide(slide, thresh, crop_size,
                                             bbox_tissue,
                                             des_folder_normal_patches,
                                             single_slide_for_patch_extraction):
    """
    The actual function for patch extraction from normal slides.

    :param slide: object generated by openslide
    :type slide: object
    :param thresh: the threshhold for tissue region
    :type thresh: list
    :param crop_size: the size of image patches to be extracted
    :type crop_size: list
    :param bbox_tissue: the bounding box for tissue region
    :type bbox_tissue: tuple
    :param des_folder_normal_patches: the folder to store the extracted patches
    :type des_folder_normal_patches: string
    :param single_slide_for_patch_extraction: the path of a WSI slide
    :type single_slide_for_patch_extraction: string
    :returns: None

    :note: The extracted image patches will be saved.

    """
    o = 0
    while o in range(0, 1000):
        nr = random_crop_normal(slide, thresh, crop_size, bbox_tissue)
        if (cv2.countNonZero(nr[1]) > crop_size[0] * crop_size[1] * 0.1):
            nmask = np.zeros((256, 256))

            saveim('%s/%s_%d_%d_N.png' % (des_folder_normal_patches,
                                          osp.splitext(osp.basename(
                                              single_slide_for_patch_extraction))[0], nr[2][0], nr[2][1]), nr[0])
            # io.imsave('/home/wli/Downloads/test/nmask/%s_%d_%d_mask.png' % (
            # osp.splitext(osp.basename(slide_paths_total[i]))[0], nr[2][0], nr[2][1]), nmask)

            # c.append(r[3])

            # zzz = testduplicates(c)
            o = o + 1


def extract_tumor_patches_from_tumor_slide(slide, ground_truth, crop_size,
                                           thresh, bbox_tumor,
                                           des_folder_tumor_patches,
                                           des_folder_tumor_patches_mask,
                                           single_slide_for_patch_extraction):
    """
    The actual function for tumor patch extraction from tumor slides.

    :param slide: object generated by openslide
    :type slide: object
    :param ground_truth: the object generated by openslide
    :type ground_truth: object
    :param crop_size: the size of image patches to be extracted
    :type crop_size: list
    :param thresh: the threshhold for tissue region
    :type thresh: list
    :param bbox_tumor: the bounding box for tumor region
    :type bbox_tumor: tuple
    :param des_folder_tumor_patches: the folder to store the extracted patches
    :param des_folder_tumor_patches_mask: the folder to store the extracted
                                          ground truth
    :param single_slide_for_patch_extraction: the path of a WSI slide
    :type single_slide_for_patch_extraction: string
    :returns: None
    :note: The extracted image patches will be saved.

    """
    m = 0
    # a = []
    while m in range(0, 1000):
        r = random_crop_tumor(slide, ground_truth,
                              thresh, crop_size, bbox_tumor)
        if (cv2.countNonZero(r[2]) > crop_size[0] * crop_size[1] * 0.5):

            saveim('%s/%s_%d_%d_T.png' % (des_folder_tumor_patches,
                                          osp.splitext(osp.basename(single_slide_for_patch_extraction))[0], r[3][0], r[3][1]), r[0])

            skio.imsave('%s/%s_%d_%d_T_mask.png' % (des_folder_tumor_patches_mask,
                                                    osp.splitext(osp.basename(single_slide_for_patch_extraction))[0], r[3][0], r[3][1]), r[2])

            # print(r[2])

            # a.append(r[3])
            # z = testduplicates(a)
            m = m + 1


def extract_normal_patches_from_tumor_slide(slide, ground_truth, crop_size, thresh, bbox_tissue, des_folder_normal_patches, single_slide_for_patch_extraction):
    """

    The actual function for normal patch extraction from tumor slides.

    :param slide: object generated by openslide
    :type slide: object
    :param ground_truth: the object generated by openslide
    :type ground_truth: object
    :param crop_size: the size of image patches to be extracted
    :type crop_size: list
    :param thresh: the threshhold for tissue region
    :type thresh: list
    :param bbox_tissue: the bounding box for tissue region
    :type bbox_tissue: tuple
    :param des_folder_normal_patches: the folder to store the extracted patches
    :type des_folder_normal_patches: string
    :param single_slide_for_patch_extraction: the path of a WSI slide
    :type single_slide_for_patch_extraction: string
    :returns: None
    :note: The extracted image patches will be saved.


    """
    n = 0
    # b=[]
    while n in range(0, 1000):
        # slide = openslide.open_slide(slide_paths[i])
        r = random_crop_tumor(slide, ground_truth, thresh,
                              crop_size, bbox_tissue)
        if (cv2.countNonZero(r[1]) > crop_size[0] * crop_size[1] * 0.1) and (cv2.countNonZero(r[2]) == 0):

            saveim('%s/%s_%d_%d_N.png' % (des_folder_normal_patches,
                                          osp.splitext(osp.basename(single_slide_for_patch_extraction))[0], r[3][0], r[3][1]), r[0])
            # io.imsave('/home/wli/Downloads/test/validation/nmask/%s_%d_%d_mask.png' % (
            # osp.splitext(osp.basename(slide_paths_total[i]))[0], r[3][0], r[3][1]), r[2])

            # b.append(r[3])
            # zz = testduplicates(b)

            n = n + 1
