#!/home/wli/env python3
# -*- coding: utf-8 -*-
"""
Title: patches near tumor
=========================

Created: 10-31-2019
Python-Version: 3.5, 3.6

Description:
------------
This is a library module used for prediction of the whole tissue area 
using slide window method with one or several pixel shift each prediction.
The basic idea is to read a bigger tissue region into memory to avoid side
effect of predicting a small 224 x 224 patch.

This module is also used by hard negative mining (main_pred_hnm_workstation
, main_pred_hnm)

"""

from tqdm import tqdm
import numpy as np
import pandas as pd
import os.path as osp
import openslide
import glob
import os
import sys
import skimage.io as io
import re
import staintools
#############################################
import stainNorm_Reinhard
import stainNorm_Macenko


def list_file_in_dir(path, file_ext):
    """
    The function is used to return a list of files in a specific directory and its subdirectories.

    :param str path: the interested directory
    :param str file_ext: file extension. for exaple, 'tif', 'jpg'

    :return: a list of files with full paths
    :rtype: list
    """
    files = [file for file in glob.glob(
        path + "**/*.%s" % file_ext, recursive=True)]
    files.sort()

    return files


def predict_batch_from_model(patches, model):
    """
    There are two values for each prediction: one is for the score of normal patches.
    ; the other one is for the score of tumor patches. The function is used to select
    the score of tumor patches
    :param array patches: a list of image patches to be predicted.
    :param object model: the trained neural network.
    :return lsit predictions: a list of scores for each predicted image patch.
                              The score here is the probability of the image as a tumor
                              image.

    """
    predictions = model.predict(patches)
    # print(predictions[:, 1])
    # print(predictions[:, 0])
    predictions = predictions[:, 1]
    return predictions


def modify_task_id(task_id, slide_category):
    """
    The task number of 150,000 is not enough. This function creates another task_id 
    to make prediction for all the slides.
    :param int origin_taskid: the task id coming from HPC
    :param string slide_category: one option from [tumor, normal, test].
    :return new taskid: an ajusted task id 
            patches_per_task: the number of patches need to be assigned to a task on HPC
            PatchNumber: a list to show the patch numbber for each slide.
    """

    if slide_category == 'tumor':
        patches_per_task = 180
        task_id = (task_id-1)*180
    elif slide_category == 'normal':
        patches_per_task = 200
        task_id = (task_id-1)*200
    elif slide_category == 'test':
        patches_per_task = 160
        task_id = (task_id-1)*160

    return (task_id, patches_per_task)

#####################################################################################################
# The following function is used to get the slide index and patch id inside a specific slide.
#####################################################################################################


def slide_patch_index(task_id, patches_per_task, PatchNumber):
    """
    This is a critical function to identify which slide and which part of slide will be predicted by
    a specific task id from HPC
    :param int task_id: the mordified task id
    :param int patches_per_task: the number of patches to be predicted per task id
    :return int i: the index for slide that will be predicted
            int j: the index for the part of slide to be predicted
    """

    SlideRange = PatchNumber.ix[(
        PatchNumber['TaskIDrange']-task_id).abs().argsort()[:2]]
    SlideIndex = SlideRange.index.tolist()

    # i is for slide index; j is for patch id
    if task_id < SlideRange.at[SlideIndex[0], 'TaskIDrange']:

        i = SlideIndex[0]

    else:

        i = SlideIndex[0] + 1

    # find the j
    if i == 0:
        j = task_id
        j_dif = 0  # j_dif will not be used if i=0
    else:

        TaskRange_previous = PatchNumber.at[PatchNumber.index[i-1], 'TaskIDrange']
        j_dif = task_id - TaskRange_previous - 1
    # important: the first several patches will not be included if only use j = j_dif

        if j_dif < patches_per_task:

            j = 0

        else:
            j = j_dif

    return (i, j, j_dif)

# what happen if a 160 range cover two slides????????
# find the WSI slide information based on i


def slide_level_param(i, index_path_pred, slide_path_pred):
    """
    The function is used to do slide level operation and return some key parameters and objects.
    :param int i: the index of slide to be predicted
    :param list index_path_pred: the list of all the index file.
    :param list slide_path_pred: the list of all WSIs to be predicted.
    :return dataframe all_samples: all the patches in a slide
            int       n_samples: the number of total patches in a slide
            object    slide: the object from OpenSlide
            string    new_slide_path: the slide path corresponding to index path
    """
    all_samples = pd.read_pickle(index_path_pred[i])

    # find the related WSI slide. new_slide_path is a list.
    new_slide_path = [x for x in slide_path_pred if re.search(
        osp.basename(index_path_pred[i]).replace('.pkl', '.tif'), x)]
    #   new_dim_path  = [x for x in dim_paths if re.search(osp.basename(index_paths[i]).replace('.pkl', '.npy'), x)]
    # new_slide_path = [x for x in slide_paths if re.search(osp.splitext(osp.basename(x))[0], index_path[i])]
    # print(new_slide_path)
    all_samples.slide_path = new_slide_path[0]

    # print(all_samples)
    n_samples = len(all_samples)
    slide = openslide.open_slide(new_slide_path[0])

    return (all_samples, n_samples, slide, new_slide_path[0])


def slide_level_info_hnm(i, index_path_pred, slide_path_pred, ground_truth_paths=False):
    """
    The function is used to do slide level operation and return some key parameters and objects.
    :param int : the index of slide to be predicted
    :param list patch_index_path: the list of all the index files.
    :param list slide_paths: the list of all WSIs to be predicted.
    :param list ground_truth_paths: the list of all ground_truth files that binary file in tif format.
    :return: dataframe all_samples: all the patches in a slide
            int       n_samples: the number of total patches in a slide
            object    slide: the object from OpenSlide
            string    new_slide_path: the slide path corresponding to index path
    """
    all_samples = pd.read_pickle(index_path_pred[i])

    # find the related WSI slide. new_slide_path is a list.
    new_slide_path = [x for x in slide_path_pred if re.search(
        osp.basename(index_path_pred[i]).replace('.pkl', '.tif'), x)]
    #   new_dim_path  = [x for x in dim_paths if re.search(osp.basename(index_paths[]).replace('.pkl', '.npy'), x)]
    # new_slide_path = [x for x in slide_paths if re.search(osp.splitext(osp.basename(x))[0], index_path[])]
    # print(new_slide_path)
    all_samples.slide_path = new_slide_path[0]

    # print(all_samples)
    n_samples = len(all_samples)
    slide = openslide.open_slide(new_slide_path[0])
    # print(n_samples)
    if ground_truth_paths:

        new_ground_truth_path = [x for x in ground_truth_paths if re.search(
            osp.basename(index_path_pred[i]).replace('.pkl', '_mask.tif'), x)]

        ground_truth = openslide.open_slide(new_ground_truth_path[0])

    return (all_samples, n_samples, slide, new_slide_path[0], ground_truth) if ground_truth_paths else (all_samples, n_samples, slide, new_slide_path[0])


####################################################################################################
# create folders to store individual prediction for patches
####################################################################################################

def creat_folder(new_slide_path, path_for_results):
    """
    To create folder to store the prediction results.
    :param string new_slide_path
    :param string path_for_results
    """
    folder_name = osp.splitext(osp.basename(new_slide_path))[0]
    path_to_create = osp.join(path_for_results, folder_name)

    try:
        os.makedirs(path_to_create)
    except:
        print('Folder exists. Skipped')

    return path_to_create


def patches_for_pred(i, j, j_dif, PatchNumber, patches_per_task, all_samples):
    """
    find the subsamples from the pkl table based on task id by using i (slide number) and j (patch number)
    :param int i: the index to find the slide to be predicted.
    :param int j: the index to find the patch series to be predicted in a slide.
    :param int patches_per_task: the number of patches to be predicted in a task on HPC.
    :return object, int subsample: a small portion of the dataframe showing the patch label and path
                        Range_right: the right end of the sample range.
    :note  Range_right is created to avoid the same task will go cross two slides.
    """
    ###
    # step1 : identify the total number of patches for slide i
    ###

    Total_Patches_One_Slide = PatchNumber.at[PatchNumber.index[i], 'PatchNumber']

    #################
    # step2: find the range of patch number inside a slide
    #####################

    #Readin_List = list(range(0, Total_Patches_One_Slide, 150))
    #Readin_List = Readin_List.extend(Total_Patches_One_Slide)

    #patch_Inlist = [x - j for x in Readin_List].argsort()[:2]

    #Patch_Index_left = Readin_List[min(Patch_Index)]

    # for test images
    if j + patches_per_task > Total_Patches_One_Slide:

        Patch_Index_right = Total_Patches_One_Slide

    else:
        # note: if has to be here to make sure the existence of Patch_index_right when i=0
        if i == 0:

            #   for test images
            Patch_Index_right = j + patches_per_task

        else:
            #       for test images
            Patch_Index_right = j_dif + patches_per_task

    # note: the last patch is Patch_Index_right -1 since iloc range does not include the end number.
    SubSamples = all_samples.iloc[j: Patch_Index_right]
    Range_Right = Patch_Index_right - 1

    return (SubSamples, Range_Right)

###################################################################################################
###################################################################################################
###################################################################################################

# all the following functions are called by the last function.
####################################################################################################
# read patches into memory for prediction
####################################################################################################
#g = 1
# create a empty list to store the results from the prediction of 150 patches


# this is read_image function from stain tools, I just copy here and use it directly since sometimes, this function from staintools
# does not work.

# def read_image(path):
#    """
#    Read an image to RGB uint8.
#    Read with opencv (cv) and covert from BGR colorspace to RGB.
#    :param path: The path to the image.
#    :return: RGB uint8 image.
#    """
#    assert os.path.isfile(path), "File not found"
#    im = cv2.imread(path)
#    # Convert from cv2 standard of BGR to our convention of RGB.
#    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#   return im
# Here we begin to set up a standard image and
# stain normalization is conducted by using vahadane's method.
####
#imagest = staintools.read_image("/home/wli/tumor_st.png")
#imagest = io.imread("/home/wli/Downloads/test/tumor_st.png")
# img = staintools.read_image(tumor_path)
#imagest_standard = standardizer.transform(imagest)
# stain_normalizer.fit(standardizer)

def color_normalization(template_image_path, color_norm_method):
    """
    The function put all the color normalization methods together.
    :param string template_image_path: the path of the image used as a template
    :param string color_norm_method: one of the three methods: vahadane, macenko, reinhard.
    :return object 
    """
    template_image = staintools.read_image(template_image_path)
    standardizer = staintools.LuminosityStandardizer.standardize(
        template_image)
    if color_norm_method == 'Reinhard':
        color_normalizer = stainNorm_Reinhard.Normalizer()
        color_normalizer.fit(standardizer)
    elif color_norm_method == 'Macenko':
        color_normalizer = stainNorm_Macenko.Normalizer()
        color_normalizer.fit(standardizer)
    elif color_norm_method == 'Vahadane':
        color_normalizer = staintools.StainNormalizer(method='vahadane')
        color_normalizer.fit(standardizer)

    return color_normalizer


def coordinates(batch_sample, pred_size):
    """
    To return the x, y coordinates for patch extraction
    :param dataframe batch_sample:  one entry from a dataframe
    :param int pred_size: the patch size to be predicted.
                          Here, the pred_size = 224
    :return list xylarge: x, y coordinates
    """
    xy = batch_sample.tile_loc[::-1]
    # double check
    if xy[0] == 0 or xy[1] == 0:
        xylarge = [x * pred_size for x in xy]
    else:
        xylarge = [x * pred_size - pred_size for x in xy]

    # print(batch_sample.tile_loc[::-1], xylarge)
    #img = tiles.get_tile(tiles.level_count-1, batch_sample.tile_loc[::-1])
    return xylarge


def generate_448_patches(slide, xylarge):
    """
    To extract 448x448 patches from WSI for prediction
    :param object slide: slide object from OpenSlide
    :param list xylarge: the x, y coordinates for the position where the patch
                          should be extracted
    :return array img: a 448x448 patch, only 3 channels (R, G, B)
    """
    img = slide.read_region(xylarge, 0, [448, 448])
    img = np.array(img)
    img = img[:, :, :3]

    return img


def generate_224_img_truth(slide, xylarge, truth=False):

    if truth:
        mask = truth.read_region(xylarge, 0, [224, 224])
        mask = np.array(mask)
        mask = mask[:, :, :3]

    else:

        mask = False

    img = slide.read_region(xylarge, 0, [224, 224])
    img = np.array(img)
    img = img[:, :, :3]

    return (img, mask)


def color_norm_pred(image_patch, fit, log_file, current_time):
    """
    To perform color normalization based on the method used.
    :param matrix img: the image to be color normalized
    :param object fit: the initialized method for normalization
    :return matrix img_norm: the normalized images
    :note if the color normalization fails, the original image patches
          will be used. But this event will be written in the log file.
    """
    img = image_patch
    img_norm = img
    try:
        img_standard = staintools.LuminosityStandardizer.standardize(img)
        img_norm = fit.transform(img_standard)
    except Exception as e:
        log_file.write(str(image_patch) + ';' + str(e) + ';' + current_time)
    # print(img_norm)
    return img_norm


def patch_pred_collect_from_slide_window(pred_size, fullimage, model, stride):
    """
    create a nxn matrix that includes all the patches extracted from one big patch by slide window sampling.

    :param integer pred_size: the size of patches to be extracted and predicted as tumor or normal patch.
    :param nxn matrix fullimage: the image used for slide window prediction, which is larger than the patch to be predicted to avoid side effect.
    :param object model: the trained network to predict the patches.

    :return a nxn matrix for one patch to be predicted by slide window method

    """

    output_preds_final = []

    for x in tqdm(range(0, pred_size, stride)):
               # print(x)
        patchforprediction_batch = []

        for y in range(0, pred_size, stride):
                       # for y in intervals:
                       # print(y)
            patchforprediction = fullimage[x:x+pred_size, y:y+pred_size]

            patchforprediction_batch.append(patchforprediction)

        X_train = np.array(patchforprediction_batch)

        preds = predict_batch_from_model(X_train, model)

        # print(preds)

        # output_preds.extend(preds)

        # output_preds_final.append(output_preds)
        output_preds_final.append(preds)

    output_preds_final = np.array(output_preds_final)

    return output_preds_final


def patch_pred_hnm(pred_size, fullimage, model, ground_truth, new_folder, new_slide_path, thresh_hold, xylarge):
    """
    create a nxn matrix that includes all the patches extracted from one big patch by slide window sampling.

    :param integer pred_size: the size of patches to be extracted and predicted as tumor or normal patch.
    :param nxn matrix fullimage: the image used for slide window prediction. The size here is equal to the pred_size = 224.
    :param object model: the trained network to predict the patches.

    :return a nxn matrix for one patch to be predicted by slide window method

    """

    X_train = np.expand_dims(fullimage, axis=0)

    pred = predict_batch_from_model(X_train, model)

    if np.count_nonzero(ground_truth):

        truth = 1

    else:

        truth = 0
        # update the dataframe with the new values
    if pred[0] > thresh_hold and truth == 0:

        io.imsave('%s/%s_%d_%d_hnm_%4.2f.png' % (new_folder,
                                                 osp.splitext(osp.basename(
                                                     new_slide_path))[0], xylarge[0], xylarge[1], pred[0]), fullimage)

    # print(fullimage)
    # print(pred[0])
    return pred[0]


def batch_pred_per_taskid(pred_size, stride, sub_samples, slide, fit, model, range_right, path_for_results, task_id, PatchNumber, i, j, current_time, log_file, color_norm=False):
    """
    This is the major function to do the prediction. It makes prediction for the image patches per task id.
    :param int pred_size: the size of image patch to be predicted. The default is 224.
    :param int stride: the pixels skipped by each slide window prediction.
    :param dataframe sub_sample: the part of dataframe with the patch location.
    :param object slide: the object from OpenSlide.
    :param object fit: the object to do color normalization
    :param object model: the trained neural network for prediction
    :param int range_right: the right end of the range of the sub_sample
    :param string folder_to_save: the location to save the prediction results.
    :param boolean color_norm: if or not the color normalization will be performed. 

    :result array of scores for each task id. For example, 160 scores for test slides.
    """
    output_preds_final_160 = []

    for g, batch_sample in sub_samples.iterrows():

        xylarge = coordinates(batch_sample, pred_size)

        if batch_sample.is_tissue == 0:
            no_tissue_region = np.zeros(
                [int(224/stride), int(224/stride)], dtype=np.float32)
            output_preds_final_160.append(no_tissue_region)

        else:

            img = generate_448_patches(slide, xylarge)

        ####################################################################################################
        # here begins the real prediction
        ####################################################################################################
            if color_norm:
                fullimage = color_norm_pred(img, fit, log_file, current_time)
            else:
                fullimage = img

            output_preds_final = patch_pred_collect_from_slide_window(
                pred_size, fullimage, model, stride)

            # print(output_preds_final)

            output_preds_final_160.append(output_preds_final)

    output_preds_final_160 = np.array(output_preds_final_160)
    # note: don't change PatchNumber to SlideRange, i can be out of range of SlideRange dataframe
    np.save(osp.join(path_for_results, '%s_%s_%d_%d' % (format(task_id, '08'),
                                                        PatchNumber.at[i, 'SlideName'], j, range_right)), output_preds_final_160)


def batch_pred_per_taskid_hnm(pred_size, stride, sub_samples, slide, fit, model, range_right, path_for_results, task_id, PatchNumber, i, j, current_time, log_file, new_slide_path, thresh_hold, truth=False, color_norm=False):
    """
    This is the major function to do the prediction. It makes prediction for the image patches per task id.
    :param int pred_size: the size of image patch to be predicted. The default is 224.
    :param int stride: the pixels skipped by each slide window prediction.
    :param dataframe sub_sample: the part of dataframe with the patch location.
    :param object slide: the object from OpenSlide.
    :param object fit: the object to do color normalization
    :param object model: the trained neural network for prediction
    :param int range_right: the right end of the range of the sub_sample
    :param string folder_to_save: the location to save the prediction results.
    :param boolean color_norm: if or not the color normalization will be performed. 

    :result array of scores for each task id. For example, 160 scores for test slides.
    """
    output_preds_final_160 = []

    for g, batch_sample in sub_samples.iterrows():

        xylarge = coordinates(batch_sample, pred_size)

        if batch_sample.is_tissue == 0:
            no_tissue_region = 0
            output_preds_final_160.append(no_tissue_region)

        else:

            img, mask = generate_224_img_truth(slide, xylarge, truth)

        ####################################################################################################
        # here begins the real prediction
        ####################################################################################################
            if color_norm:
                fullimage = color_norm_pred(img, fit, log_file, current_time)
            else:
                fullimage = img

            output_preds_final = patch_pred_hnm(
                pred_size, fullimage, model, mask, path_for_results, new_slide_path, thresh_hold, xylarge)

            # print(output_preds_final)

            output_preds_final_160.append(output_preds_final)

    output_preds_final_160 = np.array(output_preds_final_160)
    # note: don't change PatchNumber to SlideRange, i can be out of range of SlideRange dataframe
    np.save(osp.join(path_for_results, '%s_%s_%d_%d' % (format(task_id, '08'),
                                                        PatchNumber.at[i, 'SlideName'], j, range_right)), output_preds_final_160)


def exit_program(i, slide_category):
    """
    This function is only used for predicing 3d histech slides only
    :param int i: the slide index in the slide list
    :param string slide_category: can "tumor" or "normal"
    """
    if slide_category == 'normal' and (i > 99):
        sys.exit("not 3dhistech slides, program will stop")
    elif slide_category == 'tumor' and (69 < i < 109):
        sys.exit("not 3dhistech slides, program will stop")
