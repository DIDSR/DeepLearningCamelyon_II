#!/home/wli/env python3
# -*- coding: utf-8 -*-
"""
Title: wsi viewer
===========================
Created: 10-31-2019
Python-Version: 3.5, 3.6

Description:
------------
This module is used to generate overlay images to show the predition on
WSI images.

Inputs:

paths for WSI image, xml file for ground truth, heatmap

    WSI_path : the path of a WSI image;
    Xml_path : the path of a xml file, the ground truth (outline the tumor regions) of a WSI image;
    Dimension_path : the path of npy file I generated to store the dimensions of WSI image and tissue region;
    Mask_truth_path : the path of generated binary mask files from xml files;
    Heatmap_path : the path of predicted heatmap showing the scores of possible tumor region.

    slide_level = 4 : zoom level
    PPI = 150       : resolution
    threshold = 0.6 : the cutoff point for heatmap display.

Output:
*******
contours of tissue region, tumor region (from ground truth), and heatmap on WSI image

The displayed image will be saved locally. 

Request
-------

this module relys on:
         OpenSlide: https://github.com/openslide/openslide-python

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
import sys


class WSIviewer:
    '''
    The wsi_viewer class is used to display the wsi image, ground truth, predicted heatmap and their overlays

    Inputs: paths for WSI image, xml file for ground truth, heatmap

           WSI_path : the path of a WSI image;
           Xml_path : the path of a xml file, the ground truth (outline the tumor regions) of a WSI image;
           Dimension_path : the path of npy file I generated to store the dimensions of WSI image and tissue region;
           Mask_truth_path : the path of generated binary mask files from xml files;
           Heatmap_path : the path of predicted heatmap showing the scores of possible tumor region.


    Output: contours of tissue region, tumor region (from ground truth), and heatmap on WSI image


    '''
    # Class Attribute
    slide_level = 4
    PPI = 150
    threshold = 0.6

    # Initializer / Instance Attributes
    def __init__(self, WSI_path, Xml_path, Dimension_path, Mask_truth_path='', Heatmap_path=''):

        self.WSI_path = WSI_path
        self.Xml_path = Xml_path
        self.Mask_truth_path = Mask_truth_path
        self.Heatmap_path = Heatmap_path
        self.Dimension_path = Dimension_path

    # load in the files
        self.wsi_image = openslide.open_slide(WSI_path)
        self.dims = self.wsi_image.dimensions
        #ground_truth = openslide.open_slide(ground_truth_dir)
        if Mask_truth_path:

            try:
                self.mask_truth = cv2.imread(Mask_truth_path)
                self.mask_truth = self.mask_truth[:, :, 0].astype('uint8')
            except:
                self.wsi_truth = openslide.open_slide(Mask_truth_path)
                self.mask_truth_asap = self.wsi_truth.read_region((0, 0), self.slide_level, (int(
                    self.dims[0]/math.pow(2, self.slide_level)), int(self.dims[1]/math.pow(2, self.slide_level))))

        if Heatmap_path:

            self.heat_map = np.load(Heatmap_path)

        self.bbox = np.load(Dimension_path)

    # read in the wsi image at level 4, downsampled by 16
        #dims = wsi_image.level_dimensions[4]
        self.wsi_image_thumbnail = np.array(self.wsi_image.read_region((0, 0), self.slide_level, (int(
            self.dims[0]/math.pow(2, self.slide_level)), int(self.dims[1]/math.pow(2, self.slide_level)))))
        self.wsi_image_thumbnail = self.wsi_image_thumbnail[:, :, :3].astype(
            'uint8')
        self.wsi_image_thumbnail_copy = self.wsi_image_thumbnail.copy()
        self.wsi_image_thumbnail_copy_2 = self.wsi_image_thumbnail.copy()

    # read in the ground_truth

        #ground_truth_image = np.array(ground_truth.get_thumbnail((dims[0]/16, dims[1]/16)))

    def tissue_contour_on_wsi(self):

        hsv_image = cv2.cvtColor(self.wsi_image_thumbnail, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_image)
        hthresh = threshold_otsu(h)
        sthresh = threshold_otsu(s)
        #vthresh = threshold_otsu(v)
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
            self.wsi_image_thumbnail_copy, contours, -1, (0, 255, 0), 20)
        cv2.imwrite("tissue_contour_%s.png" % osp.splitext(
            osp.basename(self.WSI_path))[0], contours_on_wsi)
        return contours

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
                #X_coord = ((X_coord)*dims[0])/Ximageorg
                X_coord = X_coord/math.pow(2, self.slide_level)
                Y_coord = float(coordinate.attrib.get('Y'))
               # Y_coord = Y_coord - 155000
                #Y_coord = ((Y_coord)*dims[1])/Yimageorg
                Y_coord = Y_coord/math.pow(2, self.slide_level)
                df_xml = df_xml.append(pd.Series(
                    [Name, Order, X_coord, Y_coord], index=dfcols), ignore_index=True)  # type: DataFrame
                df_xml = pd.DataFrame(df_xml)

        return (df_xml)

    # this is a private method used for mask generation

    def _Remove(self, duplicate):
        final_list = []
        for num in duplicate:
            if num not in final_list:
                final_list.append(num)
        return final_list

    def mask_generation(self):
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

        #image = cv2.imread('/home/wli/Downloads/tumor_036.xml', -1)
        # int(self.dims[0]/math.pow(2, self.slide_level)), int(self.dims[1]/math.pow(2, self.slide_level)
        canvas = np.zeros((int(self.dims[1]/math.pow(2, self.slide_level)), int(
            self.dims[0]/math.pow(2, self.slide_level))), np.uint8)
        #tile =mr_image.getUCharPatch(0, 0, dims[0], dims[1], 4)
        # canvas = np.zeros((Ximage, Yimage, 3), np.uint8) # fix the division
        #coords = np.array([xy], dtype=np.int32)

        #cv2.drawContours(canvas, [coords],-1, (0,255,0), -1)

        #cv2.drawContours(canvas, coxy, -1, (255, 255, 255), 10)
        #cv2.drawContours(canvas, coxy, -1, (255, 255, 255), CV_FILLED)
        cv2.fillPoly(canvas, pts=coxy, color=(255, 255, 255))
        #cv2.polylines(canvas, coxy, isClosed=True, color=(255,255,255), thickness=5)

        cv2.imwrite('home_made_mask_%s.tif' % osp.splitext(
            osp.basename(self.Xml_path))[0], canvas)

        return canvas

    def truth_contour_on_wsi(self):
        # read mask file
        if self.Mask_truth_path:

            try:

                mask = self.mask_truth
                mask_binary = mask.clip(max=1)
            except:

                mask = self.mask_truth_asap
                mask_binary = np.array(mask.convert('L'))
        else:
            mask = self.mask_generation()
            mask_binary = mask.clip(max=1)

        _, contours_mask, _ = cv2.findContours(
            mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        truth_contours_wsi = cv2.drawContours(
            self.wsi_image_thumbnail, contours_mask, -1, (0, 0, 255), 20)
        cv2.imwrite("truth_contours_wsi_%s.png" % osp.basename(
            osp.splitext(self.WSI_path)[0]), truth_contours_wsi)
        return truth_contours_wsi

    def combined_contour_on_wsi(self):
        # draw contours for tissue regions and mask at the same pic
        wsi_image_thumbnail = self.wsi_image_thumbnail
        contours_tissue = self.tissue_contour_on_wsi()
        contours_mask = self.truth_contour_on_wsi().copy()
        #cv2.drawContours(wsi_image_thumbnail, contours_mask, -1, (0, 255, 0), 20)
        segmentation_mask = cv2.drawContours(
            contours_mask, contours_tissue, -1, (0, 255, 0), 20)
        plt.imshow(segmentation_mask)
        cv2.imwrite("tissue_truth_contour_%s.png" % osp.basename(
            osp.splitext(self.WSI_path)[0]), segmentation_mask)

    def display_heatmap(self):
        ''' This function is used to display the heat_map over WSI image
        input: directory of WSI image, directory of heat_map, directory of dimensions, threshhold
        output: orginal WSI image, heat_map, heat_map over WSI image,
        '''

        # make a heat_map with the same size as the downsampled wsi image
        # heatmap_final_final = np.zeros((dimension_002[0]*14, dimension_002[1]*14), pred_test_002.dtype)
        heatmap_final_final = np.zeros(
            (self.wsi_image_thumbnail.shape[0], self.wsi_image_thumbnail.shape[1]), self.heat_map.dtype)
        heatmap_final_final[self.bbox[5]*14:(self.bbox[6]+1)*14,
                            self.bbox[3]*14:(self.bbox[4]+1)*14] = self.heat_map

    # select the heat_map pixels above the threshhold
        heatmap_final_final_bin = (
            heatmap_final_final > self.threshold)*heatmap_final_final
        heatmap_final_final_bin[heatmap_final_final_bin == 0] = np.nan
        truth_contour_wsi = self.truth_contour_on_wsi()

    # display the overlay image
        plt.figure(figsize=(80, 40))
        ax1 = plt.subplot(1, 4, 1)
        ax1.imshow(self.wsi_image_thumbnail_copy_2)
        ax2 = plt.subplot(1, 4, 2)
        try:
            ax2.imshow(self.mask_truth, cmap='gray')
        except:
            ax2.imshow(self.mask_truth_asap, cmap='gray')

        ax3 = plt.subplot(1, 4, 3)
        ax3.imshow(heatmap_final_final, cmap='jet')
        ax4 = plt.subplot(1, 4, 4)
        truth_contour_wsi = self.truth_contour_on_wsi()
        ax4.imshow(truth_contour_wsi)
        ax4.imshow(heatmap_final_final_bin, cmap='jet',
                   interpolation='none', alpha=1.0, vmin=0.0, vmax=1.0)

        plt.show()

        plt.savefig('overlay_%s.png' % osp.splitext(
            osp.basename(self.WSI_path))[0], dpi=self.PPI)


if __name__ == "__main__":

    Xml_dir = '/raida/wjc/CAMELYON16/testing/lesion_annotations'
    Xml_path = '/raida/wjc/CAMELYON16/testing/lesion_annotations/tumor_026.xml'
    Xml_paths = glob.glob(osp.join(Xml_dir, '*.xml'))

    WSI_dir = '/raida/wjc/CAMELYON16/training/tumor'
    WSI_path = osp.join(WSI_dir, 'tumor_026.tif')
    WSI_paths = glob.glob(osp.join(WSI_dir, '*.tif'))

    # Mask_truth_dir = '/Users/liw17/Documents/WSI/'
    # Mask_truth_path = osp.join(Mask_truth_dir, 'test_002_truth_16.tif')
    Mask_truth_path = '/raida/wjc/CAMELYON16/training/masking/tumor_026_mask.tif'
    # Mask_truth_paths = glob.glob(osp.join(Mask_truth_dir, '*.tif'))

    # Heatmap_dir = '/Volumes/ExFAT-wzl/heat_map/test'
    # Heatmap_path = osp.join(Heatmap_dir, 'test_002.npy')
    Heatmap_path = '/raidb/wli/Final_Results/Heat_map/Method_II/color_noise_color_normalization/Method_II_Model_I_norm/tumor_0506/tumor_026.npy'
    # Heatmap_paths = glob.glob(osp.join(Heatmap_dir, '*.npy'))

    Dimension_dir = '/raidb/wli/Final_Results/Display/pred_dim_0314/testing/pred_dim_0314/testing/'
    # Dimension_path  = osp.join(Dimension_dir, 'dimensions_test_002.npy')
    Dimension_path = '//raidb/wli/Final_Results/Display/pred_dim_0314/training-updated/tumor/dimensions/tumor_026.npy'
    # Dimension_paths = glob.glob(osp.join(Dimension_dir, '*.npy'))

    viewer = WSIviewer(
        WSI_path, Xml_path, Dimension_path, Mask_truth_path, Heatmap_path)
    viewer.display_heatmap()
