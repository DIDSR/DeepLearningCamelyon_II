#!/home/wli/env python3
# -*- coding: utf-8 -*-

"""
Title: wsi visualization
=================================
Created: 10-31-2019
Python-Version: 3.5

Description:
------------
This module is used to view the WSI, its mask and heatmap overlay.

Note:
-----
The level of display resolution depends on the memory of the workstation used.
The default level is 5 (32x downsampled from the level 0 WSI image).
"""

import os.path as osp
import openslide
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pandas import DataFrame
import xml.etree.ElementTree as et
import pandas as pd
from skimage.filters import threshold_otsu
import math
import glob
import re
import os


class WSI_viewer(object):
    '''
    The wsi_viewer class is used to display the wsi image, ground truth,
    predicted heatmap and their overlays.

    :param WSI_path: the path of a WSI image
    :type WSI_path: str
    :param Xml_path: the path of a xml file, the ground truth (outline the
                      tumor regions) of a WSI image
    :type Xml_path: str
    :param Dimension_path: the path of npy file I generated to store the
                            dimensions of WSI image and tissue region.
    :type Dimension_path: str
    :param Mask_truth_path: the path of generated binary mask files from xml
                             files.
    :type Mask_truth_path: str
    :param Heatmap_path: the path of predicted heatmap showing the scores of
                          possible tumor region.
    :type Heatmap_path


    ivar: contours of tissue region, tumor region (from ground truth), and
            heatmap on WSI image.
    vartype: png

    How to Use:
    -----------

           - create an instance of the object
                   for example, viewer = WSI_viewer()
           - display the combined contours of tissue region and tumor
                   region viewer.combined_contour_on_wsi()
           - display heatmap over contours of tumor region
                    viewer.display_heatmap()
           - generate binary mask flies:
                    viewer.mask_generation()

    '''
    # Class Attribute
    slide_level = 4
    PPI = 150

    # Initializer / Instance Attributes
    def __init__(self, WSI_path, Xml_path, Dimension_path, Mask_truth_path='',
                 Heatmap_path=''):

        self.WSI_path = WSI_path
        self.Xml_path = Xml_path
        self.Mask_truth_path = Mask_truth_path
        self.Heatmap_path = Heatmap_path
        self.Dimension_path = Dimension_path

    # load in the files
        self.wsi_image = openslide.open_slide(self.WSI_path)
        # ground_truth = openslide.open_slide(ground_truth_dir)
        if self.Mask_truth_path:

            self.mask_truth = cv2.imread(self.Mask_truth_path)

        if self.Heatmap_path:
            self.heat_map = np.load(self.Heatmap_path)

        self.bbox = np.load(self.Dimension_path)

    # read in the wsi image at level 4, downsampled by 16
        self.dims = self.wsi_image.dimensions
        # dims = wsi_image.level_dimensions[4]
        self.wsi_image_thumbnail = np.array(self.wsi_image.read_region((0, 0),
                                                                       self.slide_level, (int(self.dims[0]/math.pow(2, self.slide_level)), int(self.dims[1]/math.pow(2, self.slide_level)))))
        self.wsi_image_thumbnail = self.wsi_image_thumbnail[:, :, :3].astype(
            'uint8')

    # read in the ground_truth
        self.mask_truth = self.mask_truth[:, :, 0].astype('uint8')

        # ground_truth_image = np.array(ground_truth.get_thumbnail((dims[0]/16,
        # dims[1]/16)))

        # Setter and Getter
    @property
    def WSI_path(self):
        return self.__WSI_path

    @WSI_path.setter
    def WSI_path(self, wsipath):
        if os.isdir(wsipath):
            self.__WSI_path = wsipath

    def tissue_contour_on_wsi(self, output_path):
        # read the WSI file, do not use get_thumbnail function. It has bug
        # wsi_image = openslide.open_slide(WSI_path)
        # dims = wsi_image.dimensions
        # thumbnail = wsi_image.read_region((0,0), slide_level,(int(dims[0]/32),
        # int(dims[1]/32)))
        # thumbnail = np.array(thumbnail)
        # thumbnail = thumbnail[:,:,:3]
        # thumbnail = thumbnail.astype('uint8')
        # drawcontours for tissue regions only

        hsv_image = cv2.cvtColor(self.wsi_image_thumbnail, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_image)
        hthresh = threshold_otsu(h)
        sthresh = threshold_otsu(s)
        # vthresh = threshold_otsu(v)
        # be min value for v can be changed later
        minhsv = np.array([hthresh, sthresh, 0], np.uint8)
        maxhsv = np.array([180, 255, 255], np.uint8)
        thresh = [minhsv, maxhsv]
        # extraction the countor for tissue

        rgbbinary = cv2.inRange(hsv_image, thresh[0], thresh[1])
        # plt.imshow(rgbbinary)

        rgbbinary = rgbbinary.astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        rgbbinary_close = cv2.morphologyEx(rgbbinary, cv2.MORPH_CLOSE, kernel)
        rgbbinary_open = cv2.morphologyEx(
            rgbbinary_close, cv2.MORPH_OPEN, kernel)

        _, contours, _ = cv2.findContours(
            rgbbinary_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_on_wsi = cv2.drawContours(
            self.wsi_image_thumbnail, contours, -1, (0, 255, 0), 20)
        cv2.imwrite(output_path + "%s.png" %
                    osp.splitext(osp.basename(self.WSI_path))[0], contours_on_wsi)
        return contours

    # reader = mir.MultiResolutionImageReader()
    # mr_image = reader.open('/home/wli/Downloads/tumor_036.tif')
    # Ximageorg, Yimageorg = mr_image.getDimensions()
    # dims = mr_image.getLevelDimensions(4)
    # Ximage = (Ximage+240//2)//240
    # Ximage = 4000
    # Yimage = (Yimage+240//2)//240
    # Yimage = 2000

    # this is a private method used for mask generation
    def _convert_xml_df(self):
        parseXML = et.parse(self.Xml_path)
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
                X_coord = X_coord/32
                Y_coord = float(coordinate.attrib.get('Y'))
                # Y_coord = Y_coord - 155000
                # Y_coord = ((Y_coord)*dims[1])/Yimageorg
                Y_coord = Y_coord/32
                df_xml = df_xml.append(pd.Series(
                    [Name, Order, X_coord, Y_coord], index=dfcols), ignore_index=True)  # type: DataFrame
                df_xml = pd.DataFrame(df_xml)

        return (df_xml)

    # x_values = list(annotations['X'].get_values())
    # y_values = list(annotations['Y'].get_values())
    # xy = list(zip(x_values,y_values))

    # this is a private method used for mask generation

    def _Remove(self, duplicate):
        final_list = []
        for num in duplicate:
            if num not in final_list:
                final_list.append(num)
        return final_list

    def mask_generation(self, output_path):
        # mask or ground truth generation
        annotations = self._convert_xml_df()

        final_list = self._Remove(annotations['Name'])

        # the list coxy store the x,y coordinates
        coxy = [[] for x in range(len(final_list))]

        i = 0
        for n in final_list:
            newx = annotations[annotations['Name'] == n]['X']
            newy = annotations[annotations['Name'] == n]['Y']
            print(n)
            print(newx, newy)
            newxy = list(zip(newx, newy))
            coxy[i] = np.array(newxy, dtype=np.int32)
            # note: i is different from n.
            i = i+1

        # image = cv2.imread('/home/wli/Downloads/tumor_036.xml', -1)
        canvas = np.zeros((int(self.dims[1]/math.pow(2, self.slide_level)), int(
            self.dims[0]/math.pow(2, self.slide_level))), np.uint8)
        # tile =mr_image.getUCharPatch(0, 0, dims[0], dims[1], 4)
        # canvas = np.zeros((Ximage, Yimage, 3), np.uint8) # fix the division
        # coords = np.array([xy], dtype=np.int32)

        # cv2.drawContours(canvas, [coords],-1, (0,255,0), -1)

        # cv2.drawContours(canvas, coxy, -1, (255, 255, 255), 10)
        # cv2.drawContours(canvas, coxy, -1, (255, 255, 255), CV_FILLED)
        cv2.fillPoly(canvas, pts=coxy, color=(255, 255, 255))
        # cv2.polylines(canvas, coxy, isClosed=True, color=(255,255,255), thickness=5)

        if output_path:
            cv2.imwrite(output_path + '%s.tif' %
                        osp.splitext(osp.basename(self.Xml_path))[0], canvas)

        return canvas

    def truth_contour_on_wsi(self, output_path):
        # read mask file
        if self.mask_truth:

            mask = self.mask_truth

        else:
            mask = self.mask_generation(output_path)

        mask = mask[:, :, 0]
        mask_binary = mask.clip(max=1)
        _, contours_mask, _ = cv2.findContours(
            mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        segmentation_mask = cv2.drawContours(
            self.wsi_image_thumbnail, contours_mask, -1, (0, 0, 255), 20)
        cv2.imwrite(output_path + "%s.png" %
                    osp.basename(osp.split(self.WSI_path)[0]), segmentation_mask)
        return contours_mask

    def combined_contour_on_wsi(self, output_path):
        # draw contours for tissue regions and mask at the same pic
        wsi_image_thumbnail = self.wsi_image_thumbnail
        contours_tissue = self.tissue_contour_on_wsi(output_path)
        contours_mask = self.truth_contour_on_wsi(output_path)
        cv2.drawContours(wsi_image_thumbnail,
                         contours_mask, -1, (0, 255, 0), 20)
        segmentation_mask = cv2.drawContours(
            wsi_image_thumbnail, contours_tissue, -1, (0, 0, 255), 20)

        cv2.imwrite(output_path + "%s.png" %
                    osp.basename(osp.split(self.WSI_path)[0]), segmentation_mask)

    def display_heatmap(self, threshold):
        ''' This function is used to display the heat_map over WSI image
        input: directory of WSI image, directory of heat_map, directory of dimensions, threshhold
        output: orginal WSI image, heat_map, heat_map over WSI image,
    '''

        #thumbnail_test_002 = np.array(slide.get_thumbnail(dims))
        # change to grayscale image to avoid the color confusion with heat_map.
        #wsi_image_thumbnail_grayscale = cv2.cvtColor(wsi_image_thumbnail, code=cv2.COLOR_RGB2GRAY)

    # make a heat_map with the same size as the downsampled wsi image
        #heatmap_final_final = np.zeros((dimension_002[0]*14, dimension_002[1]*14), pred_test_002.dtype)
        heatmap_final_final = np.zeros(
            (self.wsi_image_thumbnail.shape[0], self.wsi_image_thumbnail.shape[1]), self.heat_map.dtype)
        heatmap_final_final[self.bbox[5]*14:(self.bbox[6]+1)*14,
                            self.bbox[3]*14:(self.bbox[4]+1)*14] = self.heat_map

    # select the heat_map pixels above the threshhold
        heatmap_final_final_bin = (
            heatmap_final_final > threshold)*heatmap_final_final
        heatmap_final_final_bin[heatmap_final_final_bin == 0] = np.nan

    # display the overlay image
        plt.figure(figsize=(80, 40))
        ax1 = plt.subplot(1, 4, 1)
        ax1.imshow(self.wsi_image_thumbnail)
        ax2 = plt.subplot(1, 4, 2)
        ax2.imshow(self.mask_truth, cmap='gray')
        ax3 = plt.subplot(1, 4, 3)
        ax3.imshow(heatmap_final_final, cmap='jet')
        ax4 = plt.subplot(1, 4, 4)
        ax4.imshow(self.wsi_image_thumbnail)
        ax4.imshow(heatmap_final_final_bin, cmap='jet',
                   interpolation='none', alpha=1.0, vmin=0.0, vmax=1.0)

        plt.show()

        plt.savefig('overlay_%s.png' % osp.splitext(
            osp.basename(wsi_dir))[0], dpi=self.PPI)

    # the function for heatmap construction


if __name__ == "__main__":

    Threshold = 0.6

    Xml_dir = '/Users/liw17/Documents/WSI/lesion_annotations/'
    Xml_path = '/Users/liw17/Documents/WSI/lesion_annotations/tumor_026.xml'
    Xml_paths = glob.glob(osp.join(Xml_dir, '*.xml'))

    WSI_dir = '/Users/liw17/Documents/WSI/slides'
    WSI_path = osp.join(WSI_dir, '/test_002.tif')
    WSI_paths = glob.glob(osp.join(WSI_dir, '*.tif'))

    Mask_truth_dir = '/Users/liw17/Documents/WSI/'
    Mask_truth_path = osp.join(Mask_truth_dir, 'test_002_truth_16.tif')
    Mask_truth_paths = glob.glob(osp.join(Mask_truth_dir, '*.tif'))

    Heatmap_dir = '/Volumes/ExFAT-wzl/heat_map/test'
    Heatmap_path = osp.join(Heatmap_dir, 'test_002.npy')
    Heatmap_paths = glob.glob(osp.join(Heatmap_dir, '*.npy'))

    Dimension_dir = '/Users/liw17/Documents/pred_dim_0314/testing/'
    Dimension_path = osp.join(Dimension_dir, 'dimensions_test_002.npy')
    Dimension_paths = glob.glob(osp.join(Dimension_dir, '*.npy'))

    # slide path
    wsi_dir = input(
        'Enter the full directory of a WSI file including its name: ')

    print('you just input: ', wsi_dir)
    # ground_truth path
    Mask_truth_dir = input(
        'Enter the full directory of the WSI ground truth file including its name: ')

    # print('you just input: ', ground_truth_dir)
    # Heatmap Path
    Heatmap_dir = input(
        'Enter the full directory of the Heat map file including its name: ')
    print('you just input: ', Heatmap_path)
    # dimension path
    Dimensions_dir = input(
        'Enter the full directory of the dimension file including its name: ')

    # print('you just input: ', dimensions_dir)
    threshold = float(input('Enter the threshold: '))
    print('you just input: ', threshold)

    # BASE_TRUTH_DIR = Path('/home/ubuntu/data/Ground_Truth_Extracted/Mask')
    # BASE_TRUTH_DIR = Path('/Volumes/ExFAT-wzl/heat_map/test')
    # slide_path = '/home/ubuntu/data/slides/Tumor_009.tif'
    # truth_path = str(BASE_TRUTH_DIR / 'Tumor_009_Mask.tif')
    # BASE_TRUTH_DIR = Path('/Users/liw17/Downloads/camelyontest/Mask')
    # slide_path = '/Users/liw17/Documents/WSI/slides/test_002.tif'
    # pred_path = osp.join(BASE_TRUTH_DIR, 'test_002.npy')
    for wsi_image_path in WSI_dir:

        Xml_path_new = [x for x in Xml_paths if re.search(
            osp.splitext(osp.basename(wsi_image_path))[0], x)]
        Xml_path = Xml_path_new[0]
        Dimension_path_new = [x for x in Dimension_paths if re.search(
            osp.splitext(osp.basename(wsi_image_path))[0], x)]
        Dimension_path = Dimension_path[0]
        Mask_truth_path_new = [x for x in Mask_truth_paths if re.search(
            osp.splitext(osp.basename(wsi_image_path))[0], x)]
        Mask_truth_path = Mask_truth_path_new[0]
        Heatmap_path_new = [x for x in Heatmap_paths if re.search(
            osp.splitext(osp.basename(wsi_image_path))[0], x)]
        Heatmap_path = Heatmap_path_new[0]

        viewer = WSI_viewer(wsi_image_path, Xml_path, Dimension_path)
        viewer.slide_level = 5
        viewer.PPI = 150
        viewer.display_heatmap(threshold=Threshold)
