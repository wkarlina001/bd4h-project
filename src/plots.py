import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

import time
import numpy as np
import pandas as pd
import functools as ft
import seaborn as sn
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics

from src.dnn import set_global_determinism

def plot_data_distribution(df):
	# visualise data count for disease
	df_insomnia_count = (df['Insominia'].value_counts()).to_frame().reset_index()
	df_shizopherania_count = (df['shizopherania'].value_counts()).to_frame().reset_index()
	df_vascula_count = (df['vascula_demetia'].value_counts()).to_frame().reset_index()
	df_adhd_count = (df['ADHD'].value_counts()).to_frame().reset_index()
	df_bipolar_count = (df['Bipolar'].value_counts()).to_frame().reset_index()

	dfs_count = [df_insomnia_count, df_shizopherania_count, df_vascula_count, df_adhd_count, df_bipolar_count]
	df_final = ft.reduce(lambda left, right: pd.merge(left, right, on='index'), dfs_count)

	category = ['Insominia', 'shizopherania', 'vascula_demetia', 'ADHD', 'Bipolar']
	N_count_list = ((df_final.iloc[0].to_frame().reset_index()).rename(columns={'index': 'category', 0: 'value'}).iloc[1:])['value'].tolist()
	P_count_list = ((df_final.iloc[1].to_frame().reset_index()).rename(columns={'index': 'category', 1: 'value'}).iloc[1:])['value'].tolist()
	P_count_percent = []
	N_count_percent = []
	for i in range(len(P_count_list)):
		value = 100.0 * P_count_list[i] / (P_count_list[i] + N_count_list[i])
		value = round(value,2)
		P_count_percent.append(value)
		value = 100.0 * N_count_list[i] / (P_count_list[i] + N_count_list[i])
		value = round(value,2)
		N_count_percent.append(value)

	X_axis = np.arange(len(category))
	
	graph1 = plt.bar(X_axis - 0.2, N_count_list, 0.4, label = 'N')
	graph2 = plt.bar(X_axis + 0.2, P_count_list, 0.4, label = 'P')
	i = 0
	for p in graph1:
		width = p.get_width()
		height = p.get_height()
		x, y = p.get_xy()
		plt.text(x+width/2,
				y+height*1.01,
				str(round(N_count_percent[i],1))+'%',
				ha='center',
				weight='bold', size= 10)
		i+=1

	i = 0
	for p in graph2:
		width = p.get_width()
		height = p.get_height()
		x, y = p.get_xy()
		plt.text(x+width/2,
				y+height*1.01,
				str(round(P_count_percent[i],1))+'%',
				ha='center',
				weight='bold', size= 10)
		i+=1

	plt.xticks(X_axis, category)
	plt.xlabel("Disease")
	plt.ylabel("Number of Data Samples")
	plt.title("Number of Data Samples for Each Disease Category")
	plt.xticks(rotation = 45)
	plt.grid(ls='--')
	plt.legend()
	# plt.show()
	plt.tight_layout()
	plt.savefig("figure/DataSampleDistribution.png")

def plot_data_correlation(df):
	# Figure 9 correlation
	df_1 = df.drop(['Insominia', 'shizopherania', 'vascula_demetia', 'ADHD', 'Bipolar', 'target'], axis=1)
	df_1.rename(columns={'Insominia_enc': 'Insominia', 'shizopherania_enc': 'shizopherania', 'vascula_demetia_enc':'vascula_demetia', 'ADHD_enc':'ADHD', 'Bipolar_enc':'Bipolar'}, inplace=True)
	feature_names = df_1.columns.tolist()

	corrMatrix = df_1.corr()
	fig, ax = plt.subplots(figsize=(15,15))
	cmap = sn.cm.rocket_r
	sn.heatmap(corrMatrix, annot=True, cmap=cmap, xticklabels=feature_names, yticklabels=feature_names, annot_kws={"fontsize":13})
	# plt.show()
	plt.tight_layout()
	plt.savefig("figure/DataCorrelation.png")

def plot_feat_impt(X, Y, feature_category):
	# feature importance with RF
	# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

	feature_names = feature_category
	forest = RandomForestClassifier(random_state=123).fit(X, Y)

	importances = forest.feature_importances_
	std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
	forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

	fig, ax = plt.subplots()
	forest_importances.plot.barh(yerr=std, ax=ax)
	ax.set_title("Feature importances")
	ax.set_ylabel("Mean decrease in impurity")
	ax.grid(ls='--')
	fig.tight_layout()
	fig.savefig("figure/FeatureImportance.png")

def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	# pass
	plt.figure().clear()
	plt.plot(train_losses, marker='o',label ="Training Loss")
	plt.plot(valid_losses, marker='o',label ="Validation Loss")

	plt.title("Loss Curve")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend()
	plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
	plt.savefig("figure/LossCurve.png")
	
	plt.figure().clear()
	plt.plot(train_accuracies, marker='o',label ="Training Accuracy")
	plt.plot(valid_accuracies, marker='o',label ="Validation Accuracy")

	plt.title("Accuracy Curve")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.legend()
	plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
	plt.savefig("figure/AccuracyCurve.png")

def plot_roc(Y_valid, ins_pred, schiz_pred, vd_pred, adhd_pred, bp_pred):	
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = metrics.roc_curve(Y_valid[:, 0], ins_pred)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label="Insomnia (area = {:0.2f})".format(auc))
    
    fpr, tpr, _ = metrics.roc_curve(Y_valid[:, 1], schiz_pred)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label="Schizophrenia (area = {:0.2f})".format(auc))
    
    fpr, tpr, _ = metrics.roc_curve(Y_valid[:, 2], vd_pred)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label="Vascular Demetia (area = {:0.2f})".format(auc))
    
    fpr, tpr, _ = metrics.roc_curve(Y_valid[:, 3], adhd_pred)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label="ADHD (area = {:0.2f})".format(auc))
    
    fpr, tpr, _ = metrics.roc_curve(Y_valid[:, 4], bp_pred)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label="Bipolar (area = {:0.2f})".format(auc))
    
    plt.plot([0, 1], [0, 1], color = "black", linestyle='dashed')
    
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiving Operator Characteristic (ROC)")

    plt.savefig("figure/ROC.png")

def dnn_cm(X, Y, random_state):
    Y_multi = np.apply_along_axis(lambda x: ''.join(map(str, x)), 1, Y)
    Y_multi = Y_multi.reshape((-1, 1))
    ohe = OneHotEncoder(sparse=False)
    Y_multi = ohe.fit_transform(Y_multi)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y_multi, test_size=0.3, random_state=random_state)

    set_global_determinism(1234)

    model = Sequential()
    model.add(Dense(20, input_shape=(X_train.shape[1], ), activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(20, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(40, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(12, activation="softmax"))

    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=40, validation_data=(X_valid, Y_valid), verbose=0)

    pred = model.predict(X_valid, verbose = 0)
    cm = confusion_matrix(Y_valid.argmax(1), pred.argmax(1))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()

    plt.savefig("figure/DNN_CM.png")