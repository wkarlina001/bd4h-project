from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, hamming_loss, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix

def DT(X_train, X_valid, Y_train, Y_valid, SEED):
    print("Decision Tree Training and Testing..")
    dt_clf = MultiOutputClassifier(DecisionTreeClassifier(max_depth=5, random_state=SEED)).fit(X_train, Y_train)
    y_pred = dt_clf.predict(X_valid)
    y_train_pred = dt_clf.predict(X_train)
    # print('DT Model accuracy score without hyperparameter tuning on train: %.2f%%' % (accuracy_score(Y_train, y_train_pred) * 100))
    # print('DT Model accuracy score without hyperparameter tuning on valid: %.2f%%' % (accuracy_score(Y_valid, y_pred) * 100))

    param_grid = {
        "estimator__max_depth": np.arange(2,8,1),
        "estimator__min_samples_leaf": np.arange(2,10,1),
    }

    scoring = {'accuracy': make_scorer(accuracy_score),'hamming_loss':make_scorer(hamming_loss)}
    grid_cv = GridSearchCV(dt_clf, param_grid, scoring=scoring, refit="accuracy", cv=5)
    grid_cv.fit(X_train, Y_train)
    # print("DT Best parameters : ", grid_cv.best_params_)
    y_pred = grid_cv.predict(X_valid)
    y_train_pred = grid_cv.predict(X_train)
    train_accuracy = round(accuracy_score(Y_train, y_train_pred)*100, 2)
    valid_accuracy = round(accuracy_score(Y_valid, y_pred) * 100, 2)
    # print('DT Model accuracy score with hyperparameter tuning on train: %.2f%%' % (train_accuracy))
    # print('DT Model accuracy score with hyperparameter tuning on valid: %.2f%%' % (valid_accuracy))

    # print("DT Train Report : \n", classification_report(Y_train,y_train_pred))
    # print("DT Test Report : \n", classification_report(Y_valid, y_pred))
    # https://www.kaggle.com/code/kmkarakaya/multi-label-model-evaluation
    valid_multilabel_cm = (multilabel_confusion_matrix(Y_valid, y_pred))
    valid_sum_multilabel_cm = (np.sum(valid_multilabel_cm, axis=0))
    valid_multi_accuracy = round((valid_sum_multilabel_cm[0][0] + valid_sum_multilabel_cm[1][1])/ (np.sum(valid_multilabel_cm))*100, 2)

    train_multilabel_cm = (multilabel_confusion_matrix(Y_train, y_train_pred))
    train_sum_multilabel_cm = (np.sum(train_multilabel_cm, axis=0))
    train_multi_accuracy = round((train_sum_multilabel_cm[0][0] + train_sum_multilabel_cm[1][1])/ (np.sum(train_multilabel_cm))*100, 2)

    print("Decision Tree Multi-Class Accuracy - Train Accuracy %.2f%% , Validation Accuracy %.2f%%" % (train_accuracy, valid_accuracy))
    print("Decision Tree Multi-Label Accuracy - Train Accuracy %.2f%% , Validation Accuracy %.2f%%" % (train_multi_accuracy, valid_multi_accuracy))
    print("Decision Tree Multi-Label Confusion Matrix\n")
    print(valid_multilabel_cm)

    label_names = ['Insominia', 'shizopherania', 'vascula_demetia', 'ADHD', 'Bipolar']
    print(classification_report(Y_valid, y_pred,target_names=label_names))
    return train_accuracy, valid_accuracy, train_multi_accuracy, valid_multi_accuracy, train_multilabel_cm, valid_multilabel_cm

def RF(X_train, X_valid, Y_train, Y_valid, SEED):
    print("Random Forest Training and Testing..")
    clf = MultiOutputClassifier(RandomForestClassifier(max_depth=5, random_state=SEED)).fit(X_train, Y_train)
    y_pred = clf.predict(X_valid)
    y_train_pred = clf.predict(X_train)
    # print('RF Model accuracy score without hyperparameter tuning on train: %.2f%%' % (accuracy_score(Y_train, y_train_pred) * 100))
    # print('RF Model accuracy score without hyperparameter tuning on valid: %.2f%%' % (accuracy_score(Y_valid, y_pred) * 100))

    param_grid = {
        "estimator__max_depth": np.arange(2,8,1),
        "estimator__min_samples_leaf": np.arange(2,10,1),
    }

    scoring = {'accuracy': make_scorer(accuracy_score),'hamming_loss':make_scorer(hamming_loss)}
    grid_cv = GridSearchCV(clf, param_grid, scoring=scoring, refit="accuracy", cv=5)
    grid_cv.fit(X_train, Y_train)
    # print("RF Best parameters : ", grid_cv.best_params_)
    y_pred = grid_cv.predict(X_valid)
    y_train_pred = grid_cv.predict(X_train)
    train_accuracy = round(accuracy_score(Y_train, y_train_pred)*100, 2)
    valid_accuracy = round(accuracy_score(Y_valid, y_pred) * 100, 2)
    # print('RF Model accuracy score with hyperparameter tuning on train: %.2f%%' % (train_accuracy))
    # print('RF Model accuracy score with hyperparameter tuning on valid: %.2f%%' % (valid_accuracy))

    # print("RF Train Report : \n", classification_report(Y_train,y_train_pred))
    # print("RF Test Report : \n", classification_report(Y_valid, y_pred))
    valid_multilabel_cm = (multilabel_confusion_matrix(Y_valid, y_pred))
    valid_sum_multilabel_cm = (np.sum(valid_multilabel_cm, axis=0))
    valid_multi_accuracy = round((valid_sum_multilabel_cm[0][0] + valid_sum_multilabel_cm[1][1])/ (np.sum(valid_multilabel_cm))*100, 2)

    train_multilabel_cm = (multilabel_confusion_matrix(Y_train, y_train_pred))
    train_sum_multilabel_cm = (np.sum(train_multilabel_cm, axis=0))
    train_multi_accuracy = round((train_sum_multilabel_cm[0][0] + train_sum_multilabel_cm[1][1])/ (np.sum(train_multilabel_cm))*100, 2)

    print("\nRandom Forest Multi-Class Accuracy - Train Accuracy %.2f%% , Validation Accuracy %.2f%%" % (train_accuracy, valid_accuracy))
    print("Random Forest Multi-Label Accuracy - Train Accuracy %.2f%% , Validation Accuracy %.2f%%" % (train_multi_accuracy, valid_multi_accuracy))
    print("Random Forest Multi-Label Confusion Matrix\n")
    print(valid_multilabel_cm)

    label_names = ['Insominia', 'shizopherania', 'vascula_demetia', 'ADHD', 'Bipolar']
    print(classification_report(Y_valid, y_pred,target_names=label_names))
    return train_accuracy, valid_accuracy, train_multi_accuracy, valid_multi_accuracy, train_multilabel_cm, valid_multilabel_cm

def MLP(X_train, X_valid, Y_train, Y_valid, SEED):
    print("MLP Training and Testing..")
    clf = MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=(100,),batch_size=16, max_iter=300, random_state=SEED)).fit(X_train, Y_train)
    y_pred = clf.predict(X_valid)
    y_train_pred = clf.predict(X_train)
    # print('Model accuracy score without hyperparameter tuning on train: %.2f%%' % (accuracy_score(Y_train, y_train_pred) * 100))
    # print('Model accuracy score without hyperparameter tuning on valid: %.2f%%' % (accuracy_score(Y_valid, y_pred) * 100))

    lr_param = np.logspace(-4, -2, 3).tolist()
    param_grid = {
                'estimator__solver' : ['adam','sgd', 'lbfgs'],
                'estimator__activation' : ['tanh','relu'],
                'estimator__learning_rate_init' : lr_param,
                }

    scoring = {'accuracy': make_scorer(accuracy_score),'hamming_loss':make_scorer(hamming_loss)}
    grid_cv = GridSearchCV(clf, param_grid, scoring=scoring, refit="accuracy", cv=5)
    grid_cv.fit(X_train, Y_train)
    # print("MLP Best parameters : ", grid_cv.best_params_)
    y_pred = grid_cv.predict(X_valid)
    y_train_pred = grid_cv.predict(X_train)
    train_accuracy = round(accuracy_score(Y_train, y_train_pred)*100, 2)
    valid_accuracy = round(accuracy_score(Y_valid, y_pred) * 100, 2)
    # print('MLP Model accuracy score with hyperparameter tuning on train: %.2f%%' % (train_accuracy))
    # print('MLP Model accuracy score with hyperparameter tuning on valid: %.2f%%' % (valid_accuracy))

    # print("MLP Train Report : \n", classification_report(Y_train,y_train_pred))
    # print("MLP Test Report : \n", classification_report(Y_valid, y_pred))
    # plt.clf()
    # plt.plot(clf.loss_curve_)
    # plt.show()
    # plt.savefig("figure/lossCurve.png")
    valid_multilabel_cm = (multilabel_confusion_matrix(Y_valid, y_pred))
    valid_sum_multilabel_cm = (np.sum(valid_multilabel_cm, axis=0))
    valid_multi_accuracy = round((valid_sum_multilabel_cm[0][0] + valid_sum_multilabel_cm[1][1])/ (np.sum(valid_multilabel_cm))*100, 2)

    train_multilabel_cm = (multilabel_confusion_matrix(Y_train, y_train_pred))
    train_sum_multilabel_cm = (np.sum(train_multilabel_cm, axis=0))
    train_multi_accuracy = round((train_sum_multilabel_cm[0][0] + train_sum_multilabel_cm[1][1])/ (np.sum(train_multilabel_cm))*100, 2)

    print("\nMultilayer Perceptron Multi-Class Accuracy - Train Accuracy %.2f%% , Validation Accuracy %.2f%%" % (train_accuracy, valid_accuracy))
    print("\nMultilayer Perceptron Multi-Label Accuracy - Train Accuracy %.2f%% , Validation Accuracy %.2f%%" % (train_multi_accuracy, valid_multi_accuracy))
    print("Multilayer Perceptron Multi-Label Confusion Matrix\n")
    print(valid_multilabel_cm)    

    label_names = ['Insominia', 'shizopherania', 'vascula_demetia', 'ADHD', 'Bipolar']
    print(classification_report(Y_valid, y_pred,target_names=label_names))
    return train_accuracy, valid_accuracy, train_multi_accuracy, valid_multi_accuracy, train_multilabel_cm, valid_multilabel_cm

def SVM(X_train, X_valid, Y_train, Y_valid, SEED):
    print("SVM Training and Testing..")
    clf = MultiOutputClassifier(SVC(random_state=SEED)).fit(X_train, Y_train)
    y_pred = clf.predict(X_valid)
    y_train_pred = clf.predict(X_train)
    # print('SVM Model accuracy score without hyperparameter tuning on train: %.2f%%' % (accuracy_score(Y_train, y_train_pred) * 100))
    # print('SVM Model accuracy score without hyperparameter tuning on valid: %.2f%%' % (accuracy_score(Y_valid, y_pred) * 100))

    param_grid = {
        "estimator__C": np.arange(0.1,1.0,5),
        "estimator__kernel" : ["linear", "poly", "rbf", "sigmoid"]
    }

    scoring = {'accuracy': make_scorer(accuracy_score),'hamming_loss':make_scorer(hamming_loss)}
    grid_cv = GridSearchCV(clf, param_grid, scoring=scoring, refit="accuracy", cv=5)
    grid_cv.fit(X_train, Y_train)
    # print("Best parameters : ", grid_cv.best_params_)
    y_pred = grid_cv.predict(X_valid)
    y_train_pred = grid_cv.predict(X_train)
    train_accuracy = round(accuracy_score(Y_train, y_train_pred)*100, 2)
    valid_accuracy = round(accuracy_score(Y_valid, y_pred) * 100, 2)
    # print('SVM Model accuracy score with hyperparameter tuning on train: %.2f%%' % (train_accuracy))
    # print('SVM Model accuracy score with hyperparameter tuning on valid: %.2f%%' % (valid_accuracy))

    # print("SVM Train Report : \n", classification_report(Y_train,y_train_pred))
    # print("SVM Test Report : \n", classification_report(Y_valid, y_pred))
    valid_multilabel_cm = (multilabel_confusion_matrix(Y_valid, y_pred))
    valid_sum_multilabel_cm = (np.sum(valid_multilabel_cm, axis=0))
    valid_multi_accuracy = round((valid_sum_multilabel_cm[0][0] + valid_sum_multilabel_cm[1][1])/ (np.sum(valid_multilabel_cm))*100, 2)

    train_multilabel_cm = (multilabel_confusion_matrix(Y_train, y_train_pred))
    train_sum_multilabel_cm = (np.sum(train_multilabel_cm, axis=0))
    train_multi_accuracy = round((train_sum_multilabel_cm[0][0] + train_sum_multilabel_cm[1][1])/ (np.sum(train_multilabel_cm))*100, 2)

    print("\nSupport Vector Machine Multi-Class Accuracy - Train Accuracy %.2f%% , Validation Accuracy %.2f%%" % (train_accuracy, valid_accuracy))
    print("Support Vector Machine Multi-Label Accuracy - Train Accuracy %.2f%% , Validation Accuracy %.2f%%" % (train_multi_accuracy, valid_multi_accuracy))
    print("Support Vector Machine Multi-Label Confusion Matrix\n")
    print(valid_multilabel_cm)

    label_names = ['Insominia', 'shizopherania', 'vascula_demetia', 'ADHD', 'Bipolar']
    print(classification_report(Y_valid, y_pred,target_names=label_names))
    return train_accuracy, valid_accuracy, train_multi_accuracy, valid_multi_accuracy, train_multilabel_cm, valid_multilabel_cm

def SGD_SVM(X_train, X_valid, Y_train, Y_valid, SEED):
    print("SGD SVM Training and Testing..")
    clf = MultiOutputClassifier(SGDClassifier(random_state=SEED)).fit(X_train, Y_train)
    y_pred = clf.predict(X_valid)
    y_train_pred = clf.predict(X_train)
    # print('SVM Model accuracy score without hyperparameter tuning on train: %.2f%%' % (accuracy_score(Y_train, y_train_pred) * 100))
    # print('SVM Model accuracy score without hyperparameter tuning on valid: %.2f%%' % (accuracy_score(Y_valid, y_pred) * 100))

    param_grid = {
        'estimator__loss': ['hinge','log','modified_huber'],
        'estimator__average': [True, False],
        'estimator__max_iter' : np.arange(50,500,50) ,
        'estimator__alpha': np.power(10, np.arange(-5, 1, dtype=float))}

    scoring = {'accuracy': make_scorer(accuracy_score),'hamming_loss':make_scorer(hamming_loss)}
    grid_cv = GridSearchCV(clf, param_grid, scoring=scoring, refit="accuracy", cv=5)
    grid_cv.fit(X_train, Y_train)
    # print("Best parameters : ", grid_cv.best_params_)
    y_pred = grid_cv.predict(X_valid)
    y_train_pred = grid_cv.predict(X_train)
    train_accuracy = round(accuracy_score(Y_train, y_train_pred)*100, 2)
    valid_accuracy = round(accuracy_score(Y_valid, y_pred) * 100, 2)
    # print('SVM Model accuracy score with hyperparameter tuning on train: %.2f%%' % (train_accuracy))
    # print('SVM Model accuracy score with hyperparameter tuning on valid: %.2f%%' % (valid_accuracy))

    # print("SVM Train Report : \n", classification_report(Y_train,y_train_pred))
    # print("SVM Test Report : \n", classification_report(Y_valid, y_pred))
    valid_multilabel_cm = (multilabel_confusion_matrix(Y_valid, y_pred))
    valid_sum_multilabel_cm = (np.sum(valid_multilabel_cm, axis=0))
    valid_multi_accuracy = round((valid_sum_multilabel_cm[0][0] + valid_sum_multilabel_cm[1][1])/ (np.sum(valid_multilabel_cm))*100, 2)

    train_multilabel_cm = (multilabel_confusion_matrix(Y_train, y_train_pred))
    train_sum_multilabel_cm = (np.sum(train_multilabel_cm, axis=0))
    train_multi_accuracy = round((train_sum_multilabel_cm[0][0] + train_sum_multilabel_cm[1][1])/ (np.sum(train_multilabel_cm))*100, 2)

    print("\nSGD Support Vector Machine Multi-Class Accuracy - Train Accuracy %.2f%% , Validation Accuracy %.2f%%" % (train_accuracy, valid_accuracy))
    print("SGD Support Vector Machine Multi-Label Accuracy - Train Accuracy %.2f%% , Validation Accuracy %.2f%%" % (train_multi_accuracy, valid_multi_accuracy))
    print("SGD Support Vector Machine Multi-Label Confusion Matrix\n")
    print(valid_multilabel_cm)

    label_names = ['Insominia', 'shizopherania', 'vascula_demetia', 'ADHD', 'Bipolar']
    print(classification_report(Y_valid, y_pred,target_names=label_names))
    return train_accuracy, valid_accuracy, train_multi_accuracy, valid_multi_accuracy, train_multilabel_cm, valid_multilabel_cm
