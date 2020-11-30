"""
Task_remained
=============
The module is used to find the failed tasks from HPC during a variety of reasons.
The IDs of failed tasks will be stored in a file to run the makeup prediction

How to use
----------
Inputs:
*******
pred_path: the folder storing the prediction data.
slide_category: for example, 'tumor', 'normal' or 'test'
IIIdhistech_only: if only do predicion on slides from 3Dhistech scanner.

Output:
*******

result_folder

Note:
-----
This script will be run before "task_id_makeup".

"""
import numpy as np
import pandas as pd
import sys
import os
import os.path as osp
import glob
from datetime import datetime

# list folders
# for test
#path ='/scratch/weizhe.li/Pred/Method_II_Model_I_HNM/test_0414/'
#path ='/scratch/weizhe.li/Pred/Method_II_Model_II_HNM/test_0328_makeup/'

#full_path = '/scratch/weizhe.li/Pred/Method_I_Model_I_HNM/test_0407/'

# for tumor
#path ='/scratch/weizhe.li/Pred/Method_II_Model_I_HNM/tumor_0414/'
# for normal
#path ='/scratch/weizhe.li/Pred/Method_II_Model_I_HNM/normal_0414/'


def list_file_in_dir(path, file_ext):
    """
    The function is used to return a list of files in a specific directory and its subdirectories. 

    :param str path: the interested directory
    :param str file_ext: file extension. for exaple, 'tif', 'jpg'

    :return a list of files with full paths 
    """
    files = [file for file in glob.glob(
        path + "**/*.%s" % file_ext, recursive=True)]
    files.sort()

    return files


def list_file_in_dir_II(path, file_ext):
    """
    The function is used to return a list of files in a specific directory and its subdirectories. 

    :param path: the interested directory
    :type path: str
    :param file_ext: file extension. for exaple, 'tif', 'jpg'
    :type file_ext: str
    :return: a list of files with full paths 
    :rtype: list 
    """

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.%s' % file_ext in file:
                files.append(os.path.join(r, file))

    files.sort()
    print('number of npy files: %d' % len(files))
    return files


def task_id_count(files):
    """
    list all the finished task IDs
    :param files: a list all the finished tasks
    :type files: list

    :return: a list of task IDs
    :rtype: list

    """
    finished_task_id = []
    for file in files:
        # print(file)
        file = osp.basename(file)
        number_of_taskid = file[0:8]
        number_of_taskid = int(number_of_taskid)
        # print(number_of_taskid)
        finished_task_id.append(number_of_taskid)
        finished_task_id.sort()
    print('number of finished tasks: %d' % len(finished_task_id))
    return finished_task_id


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


def full_task_id(slide_category, PatchNumber, IIIdhistech_only=True):
    """
    get all the task id used for HPC

    :param slide_category: the category of slides, for example, 'tumor'
    :type slide_category: str
    :param PatchNumber: the file including the patch number for each slide in a slide category
    :type PatchNumber: dataframe
    :param IIIdhistech: for selecting only the slides from 3D histech scanner
    :type IIIdhistech: boolin

    :return task_id_full
    :rtype list

    :note this function calls another function: slide_patch_index
    """

    patches_per_task = {'tumor': 180,
                        'normal': 200,
                        'test': 160
                        }

    # The maximum number of tasks on HPC is 150000.
    full_real_predicted_taskid = list(range(0, 150000))

    full_real_predicted_taskid = [
        i * int(patches_per_task[slide_category]) for i in full_real_predicted_taskid]

    task_id_full = []
    for task_id in full_real_predicted_taskid:

        i, j, j_dif = slide_patch_index(task_id, int(
            patches_per_task[slide_category]), PatchNumber)

        if IIIdhistech_only:
            if slide_category == 'normal' and i < 101:
                task_id_full.append(task_id)
            elif slide_category == 'tumor' and (i < 71 or i == 110):
                task_id_full.append(task_id)
            elif slide_category == 'test' and i < 130:
                task_id_full.append(task_id)

    print('task_id_full : %s' % len(task_id_full))

    return task_id_full


def task_id_remain(finished_task_id, task_id_full, result_folder, result_name):
    """
    list the remained tasks needed to be done.

    :param finished_task_id: the task ID done by HPC.
    :type finished_task_id: list
    :param result_name: the name for the list to be stored locally.
    :type result_name: str

    :return: the list of task IDs needed to be done.
    :rtype: list

    :note: the list is saved as a npy file locally.
    """

    real_predicted_taskid_remain = []

    for remained_taskid in task_id_full:
        if remained_taskid not in finished_task_id:
            real_predicted_taskid_remain.append(remained_taskid)
            print(remained_taskid)

    print(len(real_predicted_taskid_remain))

    np.save('%s/%s.npy' % (result_folder, result_name),
            real_predicted_taskid_remain)

    return real_predicted_taskid_remain


if __name__ == '__main__':

    slide_categories = ['normal', 'tumor', 'test']
    slide_category = slide_categories[1]
    pred_path = '/scratch/weizhe.li/Pred_Storage/Macenko/tumor_Macenko'
    result_folder = '/home/weizhe.li/makeuptask'
    current_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    result_name = 'makeup_tumor_Macenko_1119_%s' % current_time
    IIIdhistech_only = True
    patch_numbers = {

        "normal": '/home/weizhe.li/PatchNumberForHPC_normal.pkl',
        "tumor": '/home/weizhe.li/PatchNumberForHPC_tumor.pkl',
        "test": '/home/weizhe.li/PatchNumberForHPC_test0314.pkl'
    }

    patch_number = pd.read_pickle(patch_numbers[slide_category])

    files = list_file_in_dir_II(pred_path, 'npy')
    finished_task_id = task_id_count(files)
    task_id_full = full_task_id(slide_category, patch_number, IIIdhistech_only)
    task_id_remain(finished_task_id, task_id_full, result_folder, result_name)
