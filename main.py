import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

import math
import os
from matplotlib import pyplot as plt
import functools as ft
import numpy as np
from collections import Counter
import seaborn as sn
import matplotlib.pyplot as plt

from src.dnn import *
from src.plots import *
from src.etl import *
from src.mymodels import *
import argparse

import warnings
warnings.filterwarnings("ignore")

SEED = 1234

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--smote', action='store_true')
    parser.add_argument('--no-smote', dest='smote', action='store_false')
    parser.set_defaults(smote=False)
    args = parser.parse_args()
    
    df = pd.read_csv('data/psychotic-patient-data-p2.csv')
    # data distribution for each disease category
    plot_data_distribution(df)

    # Pre Processing - clean extra characters, one hot encoding, prepare target label, filter target count < 6
    # Prepare data for training with/without SMOTE
    df, df_filter, df_target_count = clean_data(df)
    plot_data_correlation(df)

    X, Y, feature_category = prepare_data(df_target_count, df_filter, SEED, smote=args.smote)
    if args.smote is False:
        plot_feat_impt(X, Y, feature_category)

    # data preparation for modelling
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.3, random_state=SEED) 
    
    # for DNN
    print("\n Deep Neural Network Training and Testing..")
    multi_train_accuracy, multi_val_accuracy, multi_accuracies, multi_losses, _ = dnn_model(X_train, X_valid, Y_train, Y_valid, SEED, True)
    ins_train_accuracy, ins_val_accuracy, ins_accuracies, ins_losses, ins_pred = dnn_model(X_train, X_valid, Y_train, Y_valid, SEED, False, 'insomnia')
    schiz_train_accuracy, schiz_val_accuracy, schiz_accuracies, schiz_losses, schiz_pred = dnn_model(X_train, X_valid, Y_train, Y_valid, SEED, False, 'schizophrenia')
    vd_train_accuracy, vd_val_accuracy, vd_accuracies, vd_losses, vd_pred = dnn_model(X_train, X_valid, Y_train, Y_valid, SEED, False, 'vascular_demetia')
    adhd_train_accuracy, adhd_val_accuracy, adhd_accuracies, adhd_losses, adhd_pred = dnn_model(X_train, X_valid, Y_train, Y_valid, SEED, False, 'adhd')
    bp_train_accuracy, bp_val_accuracy, bp_accuracies, bp_losses, bp_pred = dnn_model(X_train, X_valid, Y_train, Y_valid, SEED, False, 'bipolar')

    print("\n\nDNN Multi-label Accuracy - Train Accuracy %.2f%% , Validation Accuracy %.2f%%" % (multi_train_accuracy*100, multi_val_accuracy*100))
    print("DNN Insomnia Accuracy - Train Accuracy %.2f%% , Validation Accuracy %.2f%%" % (ins_train_accuracy*100, ins_val_accuracy*100))
    print("DNN Schizophrenia Accuracy - Train Accuracy %.2f%% , Validation Accuracy %.2f%%" % (schiz_train_accuracy*100, schiz_val_accuracy*100))
    print("DNN Vascular Demetia Accuracy - Train Accuracy %.2f%% , Validation Accuracy %.2f%%" % (vd_train_accuracy*100, vd_val_accuracy*100))
    print("DNN ADHD Accuracy - Train Accuracy %.2f%% , Validation Accuracy %.2f%%" % (adhd_train_accuracy*100, adhd_val_accuracy*100))
    print("DNN Bipolar Accuracy - Train Accuracy %.2f%% , Validation Accuracy %.2f%%\n\n" % (bp_train_accuracy*100, bp_val_accuracy*100))
    
    # DNN plots
    plot_learning_curves(multi_losses[0], multi_losses[1], multi_accuracies[0], multi_accuracies[1])
    plot_roc(Y_valid, ins_pred, schiz_pred, vd_pred, adhd_pred, bp_pred)
    dnn_cm(X, Y, random_state=SEED)

    # # for Machine Learning train/test/tune
    dt_train_accuracy, dt_valid_accuracy, dt_train_multi_accuracy, dt_valid_multi_accuracy, dt_train_multilabel_cm, dt_valid_multilabel_cm = DT(X_train, X_valid, Y_train, Y_valid, SEED)
    rf_train_accuracy, rf_valid_accuracy, rf_train_multi_accuracy, rf_valid_multi_accuracy, rf_train_multilabel_cm, rf_valid_multilabel_cm = RF(X_train, X_valid, Y_train, Y_valid, SEED)
    svm_train_accuracy, svm_valid_accuracy, svm_train_multi_accuracy, svm_valid_multi_accuracy, svm_train_multilabel_cm, svm_valid_multilabel_cm = SVM(X_train, X_valid, Y_train, Y_valid, SEED)
    sgd_svm_train_accuracy, sgd_svm_valid_accuracy, sgd_svm_train_multi_accuracy, sgd_svm_valid_multi_accuracy, sgd_svm_train_multilabel_cm, sgd_svm_valid_multilabel_cm  = SGD_SVM(X_train, X_valid, Y_train, Y_valid, SEED)
    mlp_train_accuracy, mlp_valid_accuracy, mlp_train_multi_accuracy, mlp_valid_multi_accuracy, mlp_train_multilabel_cm, mlp_valid_multilabel_cm = MLP(X_train, X_valid, Y_train, Y_valid, SEED)

    print("\n Finish Training and Testing !")





