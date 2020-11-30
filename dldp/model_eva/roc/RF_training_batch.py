"""
Random Forest Model Training_Batch
==================================

Purpose:
========
This module is used to train a series of random forest models.

Description:
============

Inputs:
******#

 - RF_parameter_dir = '/raidb/wli/Final_Results/ROC/results/RF_parameters/RF_parameter_Method_II_Model_I_norm'

 A directory with parameters extracted from heatmaps: it has subdirectories based on the threshold. 
 In each subdirectories, there are three csv files which have the parameters extracted from the heatmap.


 - ref = pd.read_csv(
        '/raidb/wli/testing_1219/test_roc/reference_with_results_07.csv')


Output:
*******

    result_folder = '/raidb/wli/testing_1219/test_roc'

"""

import numpy as np
from sklearn.metrics import roc_curve,  precision_recall_curve, auc, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pandas as pd
import pickle
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os.path as osp
#read in files
#X_tumor = pd.read_csv('/Users/liw17/Documents/WSI/PF_parameter_MethodII_no_norm/data_sheet_for_random_forest_16_strike_tumor_9_0502_Method_II_no_norm.csv')
#X_normal = pd.read_csv('/Users/liw17/Documents/WSI/PF_parameter_MethodII_no_norm/data_sheet_for_random_forest_16_strike_normal_9_0502_Method_II_no_norm.csv')
#X_real_test = pd.read_csv('/Users/liw17/Documents/WSI/PF_parameter_MethodII_no_norm/data_sheet_for_random_forest_16_strike_test_9_0502_Method_II_no_norm.csv')
#ref = pd.read_csv('/Users/liw17/Documents/WSI/reference_new.csv')


def RF_model_training(X_tumor, X_normal):
    """
    The function is used for conducting hyper parameter search for random forest model training using grid
    search method.
    :param dataframe X_tumor: The extracted parameters from the heatmaps of tumor slides.
    :param dataframe X_normal: The extracted parameters from the heatmaps of normal slides.
    :return: the result of grind search
    """

    X = pd.concat([X_tumor, X_normal])
    X_train_unsplit = X[X.columns[3:]]

    # contruct training data for y
    y = X['tumor']

    # construct test data input
    #X_real_test = X_real_test[X_real_test.columns[3:]]

    # before train the model. split dataset
    #X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.2)

    # if do crossvalidation use the following
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_unsplit, y, test_size=0, random_state=42)

    # if scale the data to 0 - 1, we need
    #feature_scaler = StandardScaler()
    #X_train = feature_scaler.fit_transform(X_train)
    #X_test = feature_scaler.transform(X_test)
    # model training

    clf = RandomForestClassifier(
        n_estimators=300, max_features=20, max_depth=10, random_state=42)

    # to implement crossvalidation
    #all_accuracies = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=5)

    # print(all_accuracies)
    # another way to do crossvalidation
    #kfold = model_selection.KFold(n_splits=5, random_state=seed)
    #model = LogisticRegression()
    #results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    # grid search
    grid_param = {
        'n_estimators': [100, 300, 500, 800, 1000],
        'criterion': ['gini', 'entropy'],
        'bootstrap': [True, False],
        'max_depth': [5, 10, 20, 30],
        'max_features': [10, 15, 20, 25, 30],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10]
    }

    gd_sr = GridSearchCV(estimator=clf,
                         param_grid=grid_param,
                         scoring='roc_auc',
                         # scoring='accuracy',
                         cv=5,
                         n_jobs=-1)

    gd_sr.fit(X_train, y_train)

    # clf.fit(X_train,y_train)

    print('best parameter: ', gd_sr.best_params_)
    print('best score: ', gd_sr.best_score_)

    return gd_sr
#


def model_save(model, model_name, path_to_save):
    """
    Save the random forest model with best validation score. The parameters and validation score are also
    printed on the screen.
    :param model: The model is not a single model. The model here is the result from the grind search with all the parameters
                  and models.
    :return: the best model.
    :note: the best mode is saved as a pkl file.
    """
    best_parameters = model.best_params_
    print(best_parameters)

    best_result = model.best_score_
    print(best_result)

    best_model = model.best_estimator_
    # save the model

    with open('%s/%s' % (path_to_save, model_name), 'wb') as file:
        pickle.dump(best_model, file)

    return best_model

# evaluate the model


def evaluate(model, X_real_test, ground_truth, ref, folder, path_to_save):
    """
    This function is used for calculating AUC scores based on the scores predicted by the trained model and the
    ground truth.

    :param model: The trained Random Forest model.
    :type model: object, it is to be saved as a pkl file.
    :param X_real_test: the extracted parameters from heatmap for model to predict.
    :type X_real_test: dataframe
    :param ground_truth: the labels for WSIs. 0 is for normal slides; 1 is for tumor slides.
                             The ground truth is extracted from the dataframe of 'ref'.
    :type ground_truth: int
    :param ref: A dataframe read in from a CSV file with slide name and its label as ground truth.
    :type ref: dataframe
    :param folder: The directory stores the extracted parameters.
    :type folder: string
    :return: AUC score
    :rtype: float
    :note: this function also saves the ROC curve as jpg; and a dataframe with all predicted scores for WSIs
          in a CSV file.
    """
    predictions = model.predict_proba(X_real_test)[:, 1]
    roc_value = roc_auc_score(ground_truth, predictions)
    print(roc_value)
    #errors = abs(predictions - test_labels)
    #mape = 100 * np.mean(errors / test_labels)
    #accuracy = 100 - mape
    #print('Model Performance')
    #print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    #print('Accuracy = {:0.2f}%.'.format(accuracy))

    # draw roc curves
    base_fpr, base_tpr, _ = roc_curve(
        ref['truth'], [1 for _ in range(len(ref['truth']))])
    model_fpr, model_tpr, _ = roc_curve(ref['truth'], predictions)

    plt.figure(figsize=(6, 6))
    plt.rcParams['font.size'] = 16

    # Plot both curves
    # plt.plot(base_fpr, base_tpr, 'b', label='baseline')
    plt.plot(model_fpr, model_tpr, 'r', label='model')
    plt.legend()
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('1 - Specificity', fontsize=16)
    plt.ylabel('Sensitivity', fontsize=16)
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    # plt.show()
    plt.savefig('%s/overlay_224_stride_%s.png' %
                (path_to_save, folder), dpi=300)
    ref['scores_method_II_224_stride_color_norm_%s' % folder] = pd.Series(predictions)
    ref.to_csv('%s/reference_with_redo_results_224_stride_%s.csv' %
               (path_to_save, folder))

    return roc_value


def create_folder(result_folder, sub_folder):
    """
    To create folders

    :param sub_folder: the folder to be created.
    :type sub_folder: str
    :param result_folder: the folder to store the results
    :return: folder_to_create
    :rtype: str
    """

    folder_to_create = osp.join(result_folder, sub_folder)
    try:
        os.makedirs(folder_to_create)
    except:
        print('Folder exists, skip folder creation')

    return folder_to_create


if __name__ == '__main__':

    np.random.seed(42)

    #RF_parameter_dir = '/scratch/weizhe.li/RF_parameter_color_norm/stride_224'
    RF_parameter_dir = '/scratch/weizhe.li/RF_parameter_color_noise_only/stride_16'
   # RF_parameter_dir = '/scratch/weizhe.li/RF_parameter_color_without_norm'
#    RF_parameter_dir = '/raidb/wli/Final_Results/ROC/results/RF_parameters/RF_parameter_Method_II_Model_I_norm'
    ref = pd.read_csv(
        '/home/weizhe.li/reference_with_results_07.csv')
    result_folder = '/scratch/weizhe.li/RF_trained_models/RF_models_color_noise_only_with_hnm'
    sub_folder_for_models = 'stride_16'
    new_folder = create_folder(result_folder, sub_folder_for_models)

    for folder in os.listdir(RF_parameter_dir):

        # folder here represents the threshold used for parameter extraction.
        print(folder)

        folder_to_create = create_folder(new_folder, folder)

        model_name = "RFmodel_Method_I_Model_I_color_noise_only_16_stride_%s.pkl" % folder

        file_list = os.listdir(osp.join(RF_parameter_dir, folder))

        for file in file_list:

            if re.search('tumor', osp.splitext(osp.basename(file))[0]):
                X_tumor = pd.read_csv(osp.join(RF_parameter_dir, folder, file))

            elif re.search('normal', osp.splitext(osp.basename(file))[0]):
                X_normal = pd.read_csv(
                    osp.join(RF_parameter_dir, folder, file))

            elif re.search('test', osp.splitext(osp.basename(file))[0]):
                X_real_test = pd.read_csv(
                    osp.join(RF_parameter_dir, folder, file))

            else:

                print('no parameter files were found')

        print('X_tumor', X_tumor)
        print('X_normal', X_normal)
        print('X_real_test', X_real_test)

        # extract the parameters from heatmap for test dataset
        X_real_test = X_real_test[X_real_test.columns[3:]]

        # train model
        gd_model = RF_model_training(X_tumor, X_normal)

        # save model
        best_model = model_save(gd_model, model_name, folder_to_create)

        # do prediction for test dataset and calculate AUC
        roc_value_test = evaluate(
            best_model, X_real_test, ref['truth'], ref, folder, folder_to_create)