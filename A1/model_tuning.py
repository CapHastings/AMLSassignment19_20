from sklearn.base import clone
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from A1 import data_preprocess


# Select and fine-tune a Logistic Regression model.
def tune_model(X_train, y_train, X_val, y_val):
    """
    :param X_train: The predictors in training set.
    :param y_train: The labels in training set.
    :param X_val: The predictors in validation set.
    :param y_val: The labels in validation set.
    :return: The best validation-accuracy model after being fine-tuned using cross-validation.
    """
    best_acc = 0
    best_model = None
    for C in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]:
        lr = LogisticRegression(penalty='l2', C=C, solver='liblinear', max_iter=300, random_state=42)
        folds = 4  # 4-folds cross-validation.
        acc_list = []
        for i in range(folds):
            X_train_cv, y_train_cv, X_val_cv, y_val_cv = data_preprocess.create_cvset(X_train,
                                                                                      y_train,
                                                                                      X_val,
                                                                                      y_val,
                                                                                      folds,
                                                                                      i)
            lr.fit(X_train_cv, y_train_cv.ravel())
            y_pred = lr.predict(X_val_cv)
            acc_val_cv = accuracy_score(y_val_cv, y_pred)
            acc_list.append(acc_val_cv)
        acc_val = np.mean(acc_list)
        # print(acc_list, C, acc_val)
        if (best_acc < acc_val):
            best_acc = acc_val
            best_model = clone(lr)
    return best_model


# Measure the accuracy on training set.
def measure_acc_train(model, X_train, y_train):
    """
    :param model: The model with fine-tuned parameters that has not been fit on any data.
    :param X_train: The predictors in training set.
    :param y_train: The labels in training set.
    :return: The accuracy on training set.
    """
    model.fit(X_train, y_train)
    acc_train = accuracy_score(y_train, model.predict(X_train))
    return round(acc_train, 2)


# Measure the accuracy on test set.
def measure_acc_test(model, X_test, y_test):
    """
    :param model: The model with fine-tuned parameters and has been fit on training set.
    :param X_test: The predictors in test set.
    :param y_test: The labels in test set.
    :return: The accuracy on test set.
    """
    acc_test = accuracy_score(y_test, model.predict(X_test))
    return round(acc_test, 2)
