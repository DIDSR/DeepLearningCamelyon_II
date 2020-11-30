#!/home/wli/env python3
# -*- coding: utf-8 -*-
"""
Title: copy makeup files
=========================================
Created: 10-31-2019
Python-Version: 3.5
Description:
------------
This module is only used to copy the makeup files when some tasks on HPC failed,
and the makeup results were generated.

Note:
-----
This script is used after "task_id_remain" and "task_id_makeup".

"""
import numpy as np
import pandas as pd
import sys
import os
import os.path as osp
from shutil import copyfile


class copy_makeup_files(object):
    """
    :note: the makeup_path and destination_path have the same directory
    structure.
    """

    def __init__(self, makeup_path, destination_path):
        """
        for class initialization

        :param makeup_path: the folder storing the makeup results
        :type makeup_path: string
        :param destination_path: the folder storing all the results
        :type destination_path: string

        """

        self.makeup_path = makeup_path
        self.destination_path = destination_path

    def copy_file(self):

        dir = os.listdir(self.makeup_path)

        # list taskid in each folder

        makeup_file_list = []
        # full_predicted_taskid = []
        for folder in dir:
            file_list = os.listdir(osp.join(self.makeup_path, folder))
            for file in file_list:
                full_path_file = osp.join(self.makeup_path, folder, file)
                print(full_path_file)
                copyfile(full_path_file, osp.join(self.destination_path,
                                                  folder, file))
                makeup_file_list.append(full_path_file)

        return makeup_file_list


if __name__ == "__main__":

    makeup_path = '/scratch/weizhe.li/Pred_Storage/makeup_normal_Macenko'
    destination_path = '/scratch/weizhe.li/Pred_Storage/Macenko/normal_Macenko'

    makeup_copy = copy_makeup_files(makeup_path, destination_path)
    makeup_copy.copy_file()
