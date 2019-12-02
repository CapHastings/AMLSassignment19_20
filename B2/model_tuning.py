from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import numpy as np
from sklearn.metrics import accuracy_score
from B2 import data_preprocess


def tune_model(X_train, y_train, X_val, y_val):
    """
    Fine-tune a Random Forests model.

    :param X_train: The predictors in training set.
    :param y_train: The labels in training set.
    :param X_val: The predictors in validation set.
    :param y_val: The labels in validation set.
    :return: The best validation-accuracy model after being fine-tuned using cross-validation.
    """
    best_acc = 0
    best_model = None
    for n_estimators in range(50, 160, 10):
        rfc = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)
        folds = int(len(X_train) / len(X_val))
        acc_list = []
        for i in range(folds):
            X_train_cv, y_train_cv, X_val_cv, y_val_cv = data_preprocess.create_cvset(X_train,
                                                                                      y_train,
                                                                                      X_val,
                                                                                      y_val,
                                                                                      folds,
                                                                                      i)
            rfc.fit(X_train_cv, y_train_cv.ravel())
            y_pred = rfc.predict(X_val_cv)
            acc_val_cv = accuracy_score(y_val_cv, y_pred)
            acc_list.append(acc_val_cv)
        acc_val = np.mean(acc_list)
        # print(acc_list, n_estimators, acc_val)
        if (best_acc < acc_val):
            best_acc = acc_val
            best_model = clone(rfc)
    return best_model


def measure_acc_train(model, X_train, y_train):
    """
    Measure the accuracy on training set.

    :param model: The model with fine-tuned parameters that has not been fit on any data.
    :param X_train: The predictors in training set.
    :param y_train: The labels in training set.
    :return: The accuracy on training set.
    """
    model.fit(X_train, y_train)
    acc_train = accuracy_score(y_train, model.predict(X_train))
    return round(acc_train, 2)


def measure_acc_test(model, X_test, y_test):
    """
    Measure the accuracy on test set.

    :param model: The model with fine-tuned parameters and has been fit on training set.
    :param X_test: The predictors in test set.
    :param y_test: The labels in test set.
    :return: The accuracy on test set.
    """
    acc_test = accuracy_score(y_test, model.predict(X_test))
    return round(acc_test, 2)
