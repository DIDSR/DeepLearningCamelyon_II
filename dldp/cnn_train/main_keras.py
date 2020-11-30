# !/home/wli/env python3
# -*- coding: utf-8 -*-
"""
Title: training neural network
==============================

Created: 10-31-2019
Python-Version: 3.5, 3.6

Description:
------------

This module is used for training neural network.

Inputs
******

The directories of training patches:

image_patch_dirs = ['/raidb/wli/256_patches/normalized_256_patches/Vahadane']

The image_patch_dirs should be the folder includes three subfolders:

    normal_patches_from_normal_slides
    normal_patches_from_tumor_slides
    tumor_patches_from_tumor_slides

Different folder names can be used. But the labels for image patches are from
the first word of the folder names (see below for details).

The following two directories are used for hard negative mining (hnm_dir)
and a second model training (D-II) (pnt_di) :

hnm_dir = '/raidb/wli/testing_1219/hnm/tumor_baseline'

pnt_dir ='/raidb/wli/testing_1219/patch_near_tumor'

The following variables needed to be set:

"model_name" - the name of the trained model

"IIIdhistech_only = True" - The patches only from 3D histech scanner
will be used.


"training_types = ['no_color_noise']" - adding color noises or not


"folder_to_save = False" - 224x224 patches will be generated on fly.
If these patches need to be saved, set it as an abosolute path.


Outputs
*******

path_to_save_model = '/raidb/wli/testing_1219/cnn_training'

Note
----

The labels of image patches were automatically generated based on their names of
directories. The tumor image patches (labeled as "1") will be in the folder with
a name starting with "tumor_"; The normal image patches (labeled as "0") will be
in the folders with their names starting with "normal_".

For the image patches in the folders of "hard negative mining (hnm_dir)" and "patches
near tumor", they will be labeled as "0"

"""

import os
#os.environ['THEANO_FLAGS'] = "device=cuda,force_device=True,floatX=float32"
#os.environ['KERAS_BACKEND'] = 'theano'
import tensorflow
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD
from keras.initializers import RandomNormal  # Or your initializer of choice
from keras.initializers import he_normal
from keras.initializers import glorot_normal  # Or your initializer of choice
import keras.backend as K
# K.set_image_data_format('channels_first')
# os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import math
# from scipy.misc import imsave as saveim
import time
from datetime import datetime
import random
# set random seeds
#random.seed(123)
#np.random.seed(123)
#tensorflow.set_random_seed(123)
##########################################################################
# import local modules
##########################################################################

import CNN_Architecture as cnn
import Training_Patch_Pre as tpp
import dldp.utils.fileman as fm
# import googlenet_caffe as gc


exclude_normal_list = ['tumor_010', 'tumor_015', 'tumor_018', 'tumor_020',
                       'tumor_025', 'tumor_029', 'tumor_033', 'tumor_034',
                       'tumor_044', 'tumor_046', 'tumor_051', 'tumor_054',
                       'tumor_055', 'tumor_056', 'tumor_067', 'tumor_079',
                       'tumor_085', 'tumor_092', 'tumor_095', 'tumor_110']

validation_slides_normal = ['normal_003', 'normal_013', 'normal_021',
                            'normal_023', 'normal_024', 'normal_030',
                            'normal_031', 'normal_040', 'normal_045',
                            'normal_057', 'normal_062', 'normal_066',
                            'normal_068', 'normal_075', 'normal_076',
                            'normal_080', 'normal_087', 'normal_099',
                            'normal_100', 'normal_102', 'normal_106',
                            'normal_112', 'normal_117', 'normal_127',
                            'normal_132', 'normal_139', 'normal_141',
                            'normal_149', 'normal_150', 'normal_151',
                            'normal_152', 'normal_156']

validation_slides_tumor = ['tumor_002', 'tumor_008', 'tumor_010', 'tumor_019',
                           'tumor_022', 'tumor_024', 'tumor_025', 'tumor_031',
                           'tumor_040', 'tumor_045', 'tumor_049', 'tumor_069',
                           'tumor_076', 'tumor_083', 'tumor_084', 'tumor_085',
                           'tumor_088', 'tumor_091', 'tumor_101', 'tumor_102',
                           'tumor_108', 'tumor_109']


def step_decay(epoch):
    """
    The function is used to schedule the learning rate over epochs.

    :param epoch: the epoch number from network training
    :type epoch: int
    :returns: learning rate
    :rtype: float

    """
    initial_lrate = 0.01
    # drop = 0.5
    drop = 0.1
    epochs_drop = 1.0
    lrate = initial_lrate * \
        math.pow(drop, math.floor((epoch) / epochs_drop))

    return lrate


def reset_init_weight(model):

    initial_weights = model.get_weights()

    backend_name = K.backend()

    if backend_name == 'tensorflow':
        def k_eval(placeholder): return placeholder.eval(
            session=K.get_session())
    elif backend_name == 'theano':
        def k_eval(placeholder): return placeholder.eval()
    else:
        raise ValueError("Unsupported backend")

    new_weights = [k_eval(RandomNormal()(w.shape)) for w in initial_weights]

    model.set_weights(new_weights)


# class TensorBoardKeras(TensorBoard):
#     def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
#         super().__init__(log_dir=log_dir, **kwargs)

#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         logs.update({'lr': K.eval(self.model.optimizer.lr)})
#         super().on_epoch_end(epoch, logs)


def model_train(train_generator, validation_generator, model_name_path,
                model_name, sub_folder,
                BATCH_SIZE, N_EPOCHS, current_time, patches_category, class_weight_balance,
                color_noise):
    """
    The function for training neural network

    :param train_generator: the training patches per batch
    :type train_generator: array
    :param validation_generator: the training patches per batch
    :type validation_generator: array
    :param model_name_path: the place to save the model
    :type model_name_path: string
    :param model_name: the name for trained model
    :type model_name: string
    :param sub_folder: the subfolder to store the results
    :type sub_folder: string
    :param BATCH_SIZE: the batch size, here is 32
    :type BATCH_SIZE: int
    :param N_EPOCHS: the total training epochs
    :type N_EPOCHS: int
    :param current_time: the time of neural network training
    :type current_time: string
    :param patches_category: the name of the folder
                             with all training patches
    :type patches_category: string
    :param color_noise: the type of color noise
    :type color_noise: string

    :return: history
    :rtype: object

    :note: multiple trained models will be saved during training.

    """
    if color_noise:
        BATCH_SIZE = int(BATCH_SIZE/2)

    else:

        BATCH_SIZE = BATCH_SIZE
    # choose the neural network to be trained
    model = cnn.InceptionV1()
    model.summary()
    # reset_init_weight(model)
    # sgd optimizer is used here
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # compile the model
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # to save the trained model based on validation accuracy
    # model_checkpoint = ModelCheckpoint(
    # model_name_path, monitor='val_accuracy', verbose=1, save_best_only=True)
    model_checkpoint = ModelCheckpoint(
        model_name_path, monitor='val_accuracy', verbose=1, save_best_only=True)
    ######################################################################
    # begin to train the neural network
    ######################################################################
    train_start_time = datetime.now()

    learningrate = LearningRateScheduler(step_decay)

    # tensorboard = TensorBoard(log_dir='/%s/%s_%s_%s' % (sub_folder, model_name, current_time, patches_category), histogram_freq=0,
    # write_graph=True, write_images=False)

    callbacks_list = [learningrate, model_checkpoint]
    # callbacks_list = [learningrate, model_checkpoint, tensorboard]

    history = model.fit_generator(train_generator, np.ceil(
        len(training_patches) / BATCH_SIZE),
        validation_data=validation_generator,
        validation_steps=np.ceil(
        len(validation_patches) / BATCH_SIZE),
        epochs=N_EPOCHS, callbacks=callbacks_list, class_weight=class_weight_balance)

    # print (lrate)

    train_end_time = datetime.now()

    ###########################################################################
    # network training is done
    ###########################################################################
    print("Model training time: %.1f minutes" %
          ((train_end_time - train_start_time).seconds / 60,))
    # save the model.Actually the best model was saved during the training.
    # Here the model is saved again.
    model.save(
        '/%s/model_%s_%s-{epoch:02d}-{val_accuracy:.4f}.h5' % (
            sub_folder, current_time, patches_category))

    model_json = model.to_json()
    with open("/%s/%s_%s_%s-{epoch:02d}-{val_accuracy:.4f}.json" % (
            sub_folder, model_name, current_time, patches_category), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(
        "/%s/%s_%s_%s-{epoch:02d}-{val_accuracy:.4f}.h5" % (
            sub_folder, model_name, current_time, patches_category))
    print("Saving model to disk")

    return history


def model_train_caffe(train_generator, validation_generator, model_name_path,
                      model_name, sub_folder,
                      BATCH_SIZE, N_EPOCHS, current_time, patches_category, class_weight_balance,
                      color_noise):
    """
    The function for training neural network

    :param train_generator: the training patches per batch
    :type train_generator: array
    :param validation_generator: the training patches per batch
    :type validation_generator: array
    :param model_name_path: the place to save the model
    :type model_name_path: string
    :param model_name: the name for trained model
    :type model_name: string
    :param sub_folder: the subfolder to store the results
    :type sub_folder: string
    :param BATCH_SIZE: the batch size, here is 32
    :type BATCH_SIZE: int
    :param N_EPOCHS: the total training epochs
    :type N_EPOCHS: int
    :param current_time: the time of neural network training
    :type current_time: string
    :param patches_category: the name of the folder
                             with all training patches
    :type patches_category: string
    :param color_noise: the type of color noise
    :type color_noise: string

    :return: history
    :rtype: object

    :note: multiple trained models will be saved during training.

    """
    if color_noise:
        BATCH_SIZE = int(BATCH_SIZE/2)

    else:

        BATCH_SIZE = BATCH_SIZE
    # choose the neural network to be trained
    # model = cnn.InceptionV1()
    model = gc.create_googlenet()
    # reset_init_weight(model)
    # sgd optimizer is used here
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # compile the model
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # to save the trained model based on validation accuracy
    model_checkpoint = ModelCheckpoint(
        model_name_path, monitor='val_accuracy', verbose=1, save_best_only=True)
    ######################################################################
    # begin to train the neural network
    ######################################################################
    train_start_time = datetime.now()

    learningrate = LearningRateScheduler(step_decay)

    tensorboard = TensorBoard(log_dir='/%s/%s_%s_%s' % (sub_folder, model_name, current_time, patches_category), histogram_freq=0,
                              write_graph=True, write_images=False)

    # callbacks_list = [learningrate, model_checkpoint]
    callbacks_list = [learningrate, model_checkpoint, tensorboard]

    history = model.fit_generator(train_generator, np.ceil(
        len(training_patches) / BATCH_SIZE),
        validation_data=validation_generator,
        validation_steps=np.ceil(
        len(validation_patches) / BATCH_SIZE),
        epochs=N_EPOCHS, callbacks=callbacks_list, class_weight=class_weight_balance)

    # print (lrate)

    train_end_time = datetime.now()

    ###########################################################################
    # network training is done
    ###########################################################################
    print("Model training time: %.1f minutes" %
          ((train_end_time - train_start_time).seconds / 60,))
    # save the model.Actually the best model was saved during the training.
    # Here the model is saved again.
    model.save(
        '/%s/model_%s_%s-{epoch:02d}-{val_accuracy:.4f}.h5' % (
            sub_folder, current_time, patches_category))

    model_json = model.to_json()
    with open("/%s/%s_%s_%s-{epoch:02d}-{val_accuracy:.4f}.json" % (
            sub_folder, model_name, current_time, patches_category), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(
        "/%s/%s_%s_%s-{epoch:02d}-{val_accuracy:.4f}.h5" % (
            sub_folder, model_name, current_time, patches_category))
    print("Saving model to disk")

    return history


if __name__ == "__main__":
    #######################################################################
    # model training preparation
    ########
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # image_patch_dirs = ['/raidb/wli/256_patches/original_256_patches']
    image_patch_dirs = [
        '/gpfs_projects/weizhe.li/256_patches/original_256_patches']
    # Use the following folder for model training by color normalized patches.
    # image_patch_dirs = [
    #    '/raidb/wli/256_patches/normalized_256_patches/Vahadane']
    # for hard negative mining
    #hnm_dir = '/raidb/wli/testing_1219/hnm/tumor_baseline'
    # hnm_dir = '/home/wli/Documents/hnm/no_noise_no_norm'
    #hnm_dir = '/gpfs_projects/weizhe.li/HNM_patches_081420/hnm_noise_only_red_080720'
    hnm_dir = None
    # for model-II training
    # pnt_dir = '/raidb/wli/testing_1219/patch_near_tumor'
    pnt_dir = None
    crop_size = (224, 224)
    BATCH_SIZE = 32
    N_EPOCHS = 5
    current_time = time.strftime("%m.%d.%y_%H:%M", time.localtime())
    path_to_save_model = '/gpfs_projects/weizhe.li/cnn_train_0925_determ'
    model_name = 'no_noise_no_norm_102820_with_preprocess_mean_no_determ_lr0_01'

    class_weight_balance = {0: 1., 1: 1.}
    class_weight_hnm = {0: 0.33, 1: 0.67}
    class_weight_hnm_pnt = {0: 1., 1: 3.}
    # if hnm_dir:
    # class_weight_balance = class_weight_hnm
    if pnt_dir:
        class_weight_balance = class_weight_hnm_pnt
    # if you want to get the training patches from 3dhistech scanner only,
    # set IIIdhistech_only=True.
    IIIdhistech_only = None

    # This is the folder to save the 224x224 patches on fly during training.
    # Saving some of these patches is for checking them by eye.
    # folder_to_save = '/raidb/wli/augmented_patches/'
    folder_to_save = False
    training_types = ['no_color_noise']
    # training_types = ['no_color_noise', 'color_noise']
    for index, image_patch_dir in enumerate(image_patch_dirs):
        """This is the for loop to go through all
        different training dataset."""
        patches_category = '%s' % osp.split(image_patch_dir)[1]
        print("patch category: " + patches_category)

        # define the normalized patches to be used.
        for training_type in training_types:
            training_validation_patches = tpp.training_patch_paths(
                image_patch_dir, exclude_normal_list, validation_slides_normal,
                validation_slides_tumor, path_to_save_model, IIIdhistech_only, hnm_dir, pnt_dir)
            # A python dataframe is created to include the paths of
            # all the training and validation patches and their labels (tumor or not)
            training_patches = training_validation_patches[0]
            validation_patches = training_validation_patches[1]
            print('number of validation patches: %d ' %
                  len(validation_patches))
            # set the path and name of the best trained models to be saved.
            sub_folder = '%s/%s_%s' % (path_to_save_model,
                                       training_type, patches_category)

            fm.creat_folder(sub_folder)
            # os.mkdir(sub_folder)
            model_name_path = '/%s/googlenetv1_keras_no_color_no_norm_%s_%s_%s-{epoch:02d}-{val_accuracy:.4f}.hdf5' % (
                sub_folder, model_name, current_time,
                patches_category)
            if training_type == 'no_color_noise':
                train_generator = tpp.gen_imgs(
                    training_patches, BATCH_SIZE, crop_size, folder_to_save)
                validation_generator = tpp.gen_imgs(
                    validation_patches, BATCH_SIZE, crop_size, folder_to_save)
            ###################################################################
            # train the neural network
            ########
                history = model_train(train_generator, validation_generator,
                                            model_name_path,
                                            model_name, sub_folder,
                                            BATCH_SIZE, N_EPOCHS, current_time,
                                            patches_category, class_weight_balance, color_noise=False)
            ########
            # done for model training
            ###################################################################
            else:

                train_generator = tpp.gen_imgs_color_noise(
                    training_patches, BATCH_SIZE, crop_size, folder_to_save)
                validation_generator = tpp.gen_imgs(
                    validation_patches, BATCH_SIZE, crop_size, folder_to_save)
            ###################################################################
            # train the neural network
            ########
                history = model_train(train_generator, validation_generator,
                                      model_name_path,
                                      model_name, sub_folder,
                                      BATCH_SIZE, N_EPOCHS, current_time,
                                      patches_category, class_weight_balance, color_noise=True)
            ########
            # done for model training
            ###################################################################
#
            ###################################################################
            # display the training and validation scores in figures
            ########
            #  "Accuracy"
            fig1 = plt.figure(figsize=(12, 8))
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()
            fig1.savefig('/%s/Accuracy_plot_googlenet.png' % sub_folder)
            # plt.imsave('accuracy_plot_googlenet', accu)
            # plt.close(fig1)
            # "Loss"
            fig2 = plt.figure(figsize=(12, 8))
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()
            plt.savefig('/%s/Loss_plot_googlenet.png' % sub_folder)
            # plt.imsave('Loss_plot_googlenet',loss)
            # np.save('history_googlenet', history.history)

            ########
            # done for displaying training and validation scores in figures
            #######################################################################
