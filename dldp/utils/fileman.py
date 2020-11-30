#!/home/wli/env python3
# -*- coding: utf-8 -*-
"""
Title: fileman
==========================

Created: 10-31-2019
Python-Version: 3.5, 3.6

Description:
------------
This module includes functions for folder operations: folder creation,
listing the files in a folder and its subfolders.
"""


import os.path as osp
import os
import glob


def creat_folder(new_slide_path, path_for_results=''):
    """
    To create folder to store the prediction results.
    :param string new_slide_path
    :param string path_for_results
    """
    if new_slide_path:

        if osp.isfile(new_slide_path):
            folder_name = osp.splitext(osp.basename(new_slide_path))[0]
            path_to_create = osp.join(path_for_results, folder_name)
        elif os.path.isdir(new_slide_path):
            folder_name = osp.basename(new_slide_path)
            path_to_create = osp.join(path_for_results, folder_name)

        else:
            path_to_create = osp.join(path_for_results, new_slide_path)
    else:
        folder_name = ''
        path_to_create = osp.join(path_for_results, folder_name)
    try:
        os.makedirs(path_to_create)
    except Exception as e:
        print('Folder exists. Skipped' + '' + str(e))

    return path_to_create


def list_file_in_dir(path, file_ext):
    """
    The function is used to return a list of files in a specific directory and
    its subdirectories.

    :param str path: the interested directory
    :param str file_ext: file extension. for exaple, 'tif', 'jpg'

    :return a list of files with full paths
    """
    files = [file for file in glob.glob(path + "**/*.%s" % file_ext,
                                        recursive=True)]
    files.sort()

    return files


def list_file_in_dir_II(path, file_ext):
    """
    The function is used to return a list of files in a specific directory and
    its subdirectories.

    :param path: the interested directory
    :type path: str
    :param file_ext: file extension. for exaple, 'tif', 'jpg'
    :type file_ext: str

    :return: a list of files with their absolute paths
    :rtype: list

    """

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.%s' % file_ext in file:
                files.append(os.path.join(r, file))
    files.sort()
    return files
