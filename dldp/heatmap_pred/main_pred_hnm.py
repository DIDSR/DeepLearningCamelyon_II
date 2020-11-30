from keras.models import load_model
import pandas as pd
import os.path as osp
import os
import sys
from datetime import datetime
#############################################
import Pred_Slide_Window_For_Heatmap as pswh


if __name__ == "__main__":

    current_time = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
    slide_categories = ['normal', 'tumor', 'test']
    slide_category = slide_categories[0]
    batch_size = 32
    crop_size = [224, 224]
    pred_size = 224
    NUM_CLASSES = 2  # not_tumor, tumor
    stride = 224
    thresh_hold = 0.6
    origin_taskid = int(os.environ['SGE_TASK_ID'])
    color_norm_methods = ['Vahadane', 'Reinhard', 'Macenko']
    template_image_path = '/home/weizhe.li/tumor_st.png'
    color_norm = False
    if color_norm:
        color_norm_method = color_norm_methods[2]
        fit = pswh.color_normalization(template_image_path, color_norm_method)
    else:
        color_norm_method = 'baseline'
        fit = None
    path_for_results = '/scratch/weizhe.li/hnm/%s_%s' % (
        slide_category, color_norm_method)
    log_path = '/home/weizhe.li/log_files'
    log_file = open('%s/%s.txt' % (log_path, color_norm_method), 'w')
    IIIdhistech_only = False
    # the WSI images from Camelyon16 challenge.
    slide_paths = {
        "normal": '/scratch/wxc4/CAMELYON16-training/normal/',
        "tumor": '/scratch/wxc4/CAMELYON16-training/tumor/',
        "test": '/scratch/wxc4/CAMELYON16-testing/'
    }

    # the index_path is place to store all the coordinate of tiled patches
    ####################################################################################################
    # the slide and dimension information retrievaled based on the name of index_paths to make sure all
    # dimension, index_paths, slide are all matched
    ####################################################################################################
    index_paths = {
        "normal": '/home/weizhe.li/li-code4hpc/pred_dim_0314/training-updated/normal/patch_index',
        "tumor": '/home/weizhe.li/li-code4hpc/pred_dim_0314/training-updated/tumor/patch_index',
    }

    # the slide and dimension information retrievaled based on the name of index_paths to make sure all
    # dimension, index_paths, slide are all matched
    patch_numbers = {

        "normal": '/home/weizhe.li/PatchNumberForHPC_normal.pkl',
        "tumor": '/home/weizhe.li/PatchNumberForHPC_tumor.pkl',
        "test": '/home/weizhe.li/PatchNumberForHPC_test0314.pkl'
    }

    exclude_normal_list = ['tumor_010', 'tumor_015', 'tumor_018', 'tumor_020',
                           'tumor_025', 'tumor_029', 'tumor_033', 'tumor_034',
                           'tumor_044', 'tumor_046', 'tumor_051', 'tumor_054',
                           'tumor_055', 'tumor_056', 'tumor_067', 'tumor_079',
                           'tumor_085', 'tumor_092', 'tumor_095', 'tumor_110']

    ground_truth_path = False
    ground_truth_paths = False
    # collect all the information
    slide_path_pred = pswh.list_file_in_dir(slide_paths[slide_category], 'tif')
    index_path_pred = pswh.list_file_in_dir(index_paths[slide_category], 'pkl')
    patch_number = pd.read_pickle(patch_numbers[slide_category])
    if slide_category == 'tumor':
        ground_truth_paths = pswh.list_file_in_dir(ground_truth_path, 'tif')

    print(slide_path_pred)
    # load the model for prediction
    model = load_model(
        # '/home/wli/Downloads/googlenet0917-02-0.93.hdf5')
        # '/home/wli/googlenetmainmodel1119HNM-02-0.92.hdf5')
        # '/home/weizhe.li/li-code4hpc/GoogleNetV1_01_28HNM_BoarderPatches-01-0.94.hdf5')
        # '/home/weizhe.li/Training/googlenetmainmodel0227HNM_boarderpatches_noise2-00-0.94.hdf5')
        # '/home/weizhe.li/Training/googlenetmainmodel0227HNM_noise-01-0.93.hdf5')
        # '/home/weizhe.li/Training/googlenetmainmodel0430HNM_noise-04-0.92.hdf5')#Method_II_Model_I_color_noise_and_color_norm
        # '/home/weizhe.li/Training/googlenetmainmodel0512HNM_noise_boarder-01-0.92.hdf5') #Method_II_Model_II_color_noise_and_color_norm
        # '/home/weizhe.li/Trained_Models/Reinhard/googlenetmainmodel0923_3dhistech_reinhard-02-0.90.hdf5') #3dhistech_reinhard
        # '/home/weizhe.li/Trained_Models/Macenko/googlenetmainmodel0826_3dhistech_macenko-02-0.90.hdf5'
        # '/home/weizhe.li/Trained_Models/Vahadane/googlenetmainmodel0910_3dhistech_Vahadane-03-0.90.hdf5'
        #'/home/weizhe.li/Trained_Models/googlenetv1_color_norm_r_m_10.10.19 01:59_Reinhard-05-0.90.hdf5'
        #'/home/weizhe.li/Trained_Models/googlenetv1_color_norm_r_m_10.10.19 01:59_Macenko-04-0.89.hdf5'
        #'/home/weizhe.li/Trained_Models/googlenetmainmodel_09.23.19 02:24_3dhistech_Baseline-03-0.92.hdf5'
        '/home/weizhe.li/Training/googlenetv1_Total_Patch_Retrain_10.04.19 09:06_Origin-06-0.95.hdf5')
    # modify task_id and decide the number of patches to be predicted for one task id from HPC
    task_id, patches_per_task = pswh.modify_task_id(
        origin_taskid, slide_category)
    # identify the slide and patches index
    i, j, j_dif = pswh.slide_patch_index(
        task_id, patches_per_task, patch_number)
    # select slides from 3dhistech scanner
    if IIIdhistech_only:
        pswh.exit_program(i, slide_category)

    if ground_truth_paths:
        all_samples, n_samples, slide, new_slide_path, ground_truth = pswh.slide_level_info_hnm(
            i, index_path_pred, slide_path_pred, ground_truth_paths)
    else:
        all_samples, n_samples, slide, new_slide_path = pswh.slide_level_info_hnm(
            i, index_path_pred, slide_path_pred)

    if osp.splitext(osp.basename(new_slide_path))[0] in exclude_normal_list:
        sys.exit("Not fully annotated WSI, Skip")

    path_to_create = pswh.creat_folder(new_slide_path, path_for_results)

    sub_samples, range_right = pswh.patches_for_pred(
        i, j, j_dif, patch_number, patches_per_task, all_samples)

    pswh.batch_pred_per_taskid_hnm(pred_size, stride, sub_samples, slide, fit, model, range_right, path_to_create, task_id,
                                   patch_number, i, j, current_time, log_file, new_slide_path, thresh_hold, ground_truth=False, color_norm=False)
