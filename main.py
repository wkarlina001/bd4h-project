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

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset
import torch.optim as optim

from src.utils import train, evaluate
from src.plots import *
from src.etl import *
from src.mymodels import *
import argparse

SEED = 123

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

    # for Machine Learning train/test/tune
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.3, random_state=SEED)
    dt_train_accuracy, dt_valid_accuracy = DT(X_train, X_valid, Y_train, Y_valid, SEED)
    rf_train_accuracy, rf_valid_accuracy = RF(X_train, X_valid, Y_train, Y_valid, SEED)
    svm_train_accuracy, svm_valid_accuracy = SVM(X_train, X_valid, Y_train, Y_valid, SEED)
    mlp_train_accuracy, mlp_valid_accuracy = MLP(X_train, X_valid, Y_train, Y_valid, SEED)

    # For DNN 
    
