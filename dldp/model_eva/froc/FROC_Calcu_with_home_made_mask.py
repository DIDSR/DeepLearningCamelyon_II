# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:09:32 2016
@author: Babak Ehteshami Bejnordi
Evaluation code for the Camelyon16 challenge on cancer metastases detection.

Description by Weizhe Li:
=========================

    The functions:  computeEvaluationMask
                    computeITCList

                    These two functions are used to extract the ground
                    truth from mask file.

    The function:   readCSVContent

                    The function is used to read the CSV files for each
                    WSI image in a result folder

    The function:   compute_FP_TP_Probs

                    The main function to get FP and TP

    The functions:  computeFROC
                    plotFROC

                    The functions are used to get FROC data points on
                    the FROC curve.

Note:
=====

Please remove the result for test_114.
"""

import openslide
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from skimage import measure
import os
import sys
import pandas as pd
import cv2 as cv2
import csv
from sklearn import metrics


# the following two functions are used to generated the information about each lesion from ground truth mask file. the mask file has tumor pixel as 255, normal pixel as 0.

def computeEvaluationMask(maskDIR, resolution, level):
    """Computes the evaluation mask.

    Args:
        maskDIR:    the directory of the ground truth mask
        resolution: Pixel resolution of the image at level 0
        level:      The level at which the evaluation mask is made

    Returns:
        evaluation_mask
    """
    #slide = openslide.open_slide(maskDIR)
    #dims = slide.level_dimensions[level]
    #pixelarray = np.zeros(dims[0] * dims[1], dtype='uint')
    #pixelarray = np.array(slide.read_region((0, 0), level, dims))
    # note: here 255 is used to minus pixelarray, it means the ground truth is 255 for tumor.
    # the mask file I generated is 1 for tumor region, so I changed to 1.
    pixelarray = cv2.imread(maskDIR)
    distance = nd.distance_transform_edt(255 - pixelarray[:, :, 0])
    # 75µm is the equivalent size of 5 tumor cells
    Threshold = 75 / (resolution * pow(2, level) * 2)
    binary = distance < Threshold
    filled_image = nd.morphology.binary_fill_holes(binary)
    evaluation_mask = measure.label(filled_image, connectivity=2)
    max_label = np.amax(evaluation_mask)
    print('max label = ', max_label)

    return evaluation_mask


def computeITCList(evaluation_mask, resolution, level):
    """Compute the list of labels containing Isolated Tumor Cells (ITC)

    Description:
        A region is considered ITC if its longest diameter is below 200µm.
        As we expanded the annotations by 75µm, the major axis of the object
        should be less than 275µm to be considered as ITC (Each pixel is
        0.243µm*0.243µm in level 0). Therefore the major axis of the object
        in level 5 should be less than 275/(2^5*0.243) = 35.36 pixels.

    Args:
        evaluation_mask:    The evaluation mask
        resolution:         Pixel resolution of the image at level 0
        level:              The level at which the evaluation mask was made

    Returns:
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
    """
    max_label = np.amax(evaluation_mask)
    # regionprops here is used to get the measurement of the axis_length from the ground truth lesions.
    properties = measure.regionprops(evaluation_mask)
    Isolated_Tumor_Cells = []
    threshold = 275 / (resolution * pow(2, level))
    for i in range(0, max_label):
        if properties[i].major_axis_length < threshold:
            Isolated_Tumor_Cells.append(i + 1)
    return Isolated_Tumor_Cells


# from here and the following, the code begins to

def readCSVContent(csvDIR):
    """Reads the data inside CSV file

    Args:
        csvDIR:    The directory including all the .csv files containing the results.
        Note that the CSV files should have the same name as the original image

    Returns:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
    """
    Xcorr, Ycorr, Probs = ([] for i in range(3))
    # I made a change here. the original code also read in the headline

    # csv_lines = open(csvDIR,"r").readlines()
    csv_lines = open(csvDIR, "r").readlines()[1:]
    for i in range(len(csv_lines)):
        line = csv_lines[i]
        elems = line.rstrip().split(',')
        # my csv file has an addtional column to show the index. So I begin at 1 not 0 here.
        Probs.append(float(elems[0 + 1]))
        Xcorr.append(int(float(elems[1 + 1])/4))
        # Xcorr.append(int(elems[1+1]))
        Ycorr.append(int(float(elems[2 + 1])/4))
        # Ycorr.append(int(elems[2+1]))
        print(Xcorr, Ycorr)
    return Probs, Xcorr, Ycorr


def compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, Isolated_Tumor_Cells, level):
    """Generates true positive and false positive stats for the analyzed image

    Args:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
        is_tumor:   A boolean variable which is one when the case cotains tumor
        evaluation_mask:    The evaluation mask
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
        level:      The level at which the evaluation mask was made

    Returns:
        FP_probs:   A list containing the probabilities of the false positive detections

        TP_probs:   A list containing the probabilities of the True positive detections

        NumberOfTumors: Number of Tumors in the image (excluding Isolate Tumor Cells)

        detection_summary:   A python dictionary object with keys that are the labels
        of the lesions that should be detected (non-ITC tumors) and values
        that contain detection details [confidence score, X-coordinate, Y-coordinate].
        Lesions that are missed by the algorithm have an empty value.

        FP_summary:   A python dictionary object with keys that represent the
        false positive finding number and values that contain detection
        details [confidence score, X-coordinate, Y-coordinate].
    """

    max_label = np.amax(evaluation_mask)
    FP_probs = []
    TP_probs = np.zeros((max_label,), dtype=np.float32)
    detection_summary = {}
    FP_summary = {}
    for i in range(1, max_label + 1):
        if i not in Isolated_Tumor_Cells:
            label = 'Label ' + str(i)
            detection_summary[label] = []

    FP_counter = 0
    if (is_tumor):
        for i in range(0, len(Xcorr)):
            # note: the x, y coordinates are switched, I make the x, y to be int, so that the array of evaluation_mask
            #HittedLabel = evaluation_mask[int(Xcorr[i] / pow(2, level)), int(Ycorr[i] / pow(2, level))]

            HittedLabel = evaluation_mask[int(
                Ycorr[i]/pow(2, level)), int(Xcorr[i]/pow(2, level))]
            print(HittedLabel)

            # HittedLabel = evaluation_mask[int(Ycorr[i]/pow(2, level)), int(Xcorr[i]/pow(2, level))]
            # HittedLabel = evaluation_mask[Ycorr[i]/pow(2, level), Xcorr[i]/pow(2, level)]
            if HittedLabel == 0:
                FP_probs.append(Probs[i])
                key = 'FP ' + str(FP_counter)
                FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]]
                FP_counter += 1
            elif HittedLabel not in Isolated_Tumor_Cells:
                if (Probs[i] > TP_probs[HittedLabel - 1]):
                    label = 'Label ' + str(HittedLabel)
                    detection_summary[label] = [Probs[i], Xcorr[i], Ycorr[i]]
                    TP_probs[HittedLabel - 1] = Probs[i]
    else:
        for i in range(0, len(Xcorr)):
            FP_probs.append(Probs[i])
            key = 'FP ' + str(FP_counter)
            FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]]
            FP_counter += 1

        print(FP_counter)

    num_of_tumors = max_label - len(Isolated_Tumor_Cells)
    # just for diagnose
    print('number of isolated tumor cells =', len(Isolated_Tumor_Cells))
    return FP_probs, TP_probs, num_of_tumors, detection_summary, FP_summary


def computeFROC(FROC_data):
    """Generates the data required for plotting the FROC curve

    Args:
        FROC_data:      Contains the list of TPs, FPs, number of tumors in each image

    Returns:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds

        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds
    """

    unlisted_FPs = [item for sublist in FROC_data[1] for item in sublist]
    unlisted_TPs = [item for sublist in FROC_data[2] for item in sublist]

    total_FPs, total_TPs = [], []
    all_probs = sorted(set(unlisted_FPs + unlisted_TPs))
    for Thresh in all_probs[1:]:
        total_FPs.append((np.asarray(unlisted_FPs) >= Thresh).sum())
        total_TPs.append((np.asarray(unlisted_TPs) >= Thresh).sum())
    total_FPs.append(0)
    total_TPs.append(0)
    total_FPs = np.asarray(total_FPs) / float(len(FROC_data[0]))
    total_sensitivity = np.asarray(total_TPs) / float(sum(FROC_data[3]))
    return total_FPs, total_sensitivity


def plotFROC(total_FPs, total_sensitivity):
    """Plots the FROC curve

    Args:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds

        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds

    Returns:
        -
    """
    fig = plt.figure()
    plt.xlabel('Average Number of False Positives', fontsize=12)
    plt.ylabel('Metastasis detection sensitivity', fontsize=12)
    fig.suptitle(
        'Free response receiver operating characteristic curve', fontsize=12)
    plt.plot(total_FPs, total_sensitivity, '-', color='#000000')
    plt.show()


if __name__ == "__main__":
    # The home made mask file
    mask_folder = '/raidb/wli/Final_Results/Display/home_made_mask_files_32xdownsample'
    #mask_folder = '/home/wzli/Downloads/CAMELYON16/testing/test_masking'
    # mask_folder = "...\\Camelyon16\\Ground_Truth\\Masks"
    # The folder has the CSV files
    # result_folder = '/raidb/wli/Final_Results/FROC/FROC_results/FROC_test_Method_II_noise_norm'
    result_folder = '/raidb/wli/testing_1219/test_froc/lesion_coordinates_16_stride'

    # result_folder = "...\\Camelyon16\\Results"
    result_file_list = []
    result_file_list += [each for each in os.listdir(
        result_folder) if each.endswith('.csv')]
    # I sort the file name here so that it begin at 001.file instead of random file
    result_file_list.sort()
    # The file for ground truth
    ref = pd.read_csv('/raidb/wli/testing_1219/test_froc/reference_new.csv')
    EVALUATION_MASK_LEVEL = 5  # Image level at which the evaluation is done
    # EVALUATION_MASK_LEVEL = 0 # Image level at which the evaluation is done
    L0_RESOLUTION = 0.243  # pixel resolution at level 0
    # FROC_data is an important array to store the FP, sensitivity information. It has four columns
    FROC_data = np.zeros((4, len(result_file_list)), dtype=np.object)
    FP_summary = np.zeros((2, len(result_file_list)), dtype=np.object)
    detection_summary = np.zeros((2, len(result_file_list)), dtype=np.object)
    # caseNum is the count for file number in the result folder, every for cycle, it added by 1 and represent the row of three tables here: FROC_data, FP_summary, detection_summary.
    caseNum = 0
    # the variable case is a string storing the file name.
    for case in result_file_list:
        print('Evaluating Performance on image:',
              case[0:-4])  # print out the image name but not including the extend like .tif
        sys.stdout.flush()
        csvDIR = os.path.join(result_folder, case)
        # read in the CSV file from the result_folder
        Probs, Xcorr, Ycorr = readCSVContent(csvDIR)

        # find the row number with value == slide_name
        ref_row = int(ref[ref['slide_name'] == case[0:-4]].index[0])
        is_tumor = ref['truth'].at[ref_row]
        # is_tumor = case[
        #          0:5] == 'tumor'  # this is for testing if a file begin with the word 'tumor'. I used this one since my file name starts with low case letter.
        # is_tumor = case[0:5] == 'Tumor'   # this is for testing if a file begin with the word 'Tumor'
        # according to the files in the result folder, find the corresponding mask file which end with Mask.tif
        # if it is a tumor file, do the calculation; if it is normal, there is no point to do the calculation.
        if (is_tumor):
            maskDIR = os.path.join(mask_folder, case[0:-4]) + '.tif'
            #maskDIR = os.path.join(mask_folder, case[0:-4]) + '_mask.tif'
            # maskDIR = os.path.join(mask_folder, case[0:-4]) + '_Mask.tif'
            evaluation_mask = computeEvaluationMask(
                maskDIR, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
            ITC_labels = computeITCList(
                evaluation_mask, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
        else:
            evaluation_mask = 0
            ITC_labels = []

        # the first column is the name of file for all the three table.
        FROC_data[0][caseNum] = case
        FP_summary[0][caseNum] = case
        detection_summary[0][caseNum] = case
        # The following function build up three tables based on the results from compute_FP_TP_Probs, which returns FP_probs, TP_probs, num_of_tumors, detection_summary, FP_summar
        FROC_data[1][caseNum], FROC_data[2][caseNum], FROC_data[3][caseNum], detection_summary[1][caseNum], \
            FP_summary[1][caseNum] = compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, ITC_labels,
                                                         EVALUATION_MASK_LEVEL)
        caseNum += 1

    # Compute FROC curve based on the table FROC_data created in the above for loop. this table include all the prediction of WSI files.
    total_FPs, total_sensitivity = computeFROC(FROC_data)
    #np.save('/home/wli/Downloads/FROC_data.npy', FROC_data)
    #np.save('/home/wli/Downloads/FP_summary.npy', FP_summary)
    #np.save('/home/wli/Downloads/detection_summary.npy', detection_summary)
    #np.save('/home/wli/Downloads/total_FPs_sensitivity.npy', zip(total_FPs, total_sensitivity))
    # Save the data points from FROC curve to a CSV file
    with open('/raidb/wli/testing_1219/test_froc/test_FP_sen_method_II_noise_norm_16_stride.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(total_sensitivity, total_FPs))
    # plot FROC curve
    # print(metrics.auc(total_sensitivity, total_FPs))

    plotFROC(total_FPs, total_sensitivity)
