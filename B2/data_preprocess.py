from PIL import Image
import pandas as pd
import numpy as np


def resize_image(resized_width, img):
    """
    Resize the width of an image and its height in proportion.

    :param resized_width: The width of an image after being resized.
    :param img: An Image object.
    :return: The Image object resized.
    """
    percent = (resized_width / float(img.size[0]))
    resized_height = int((float(img.size[1]) * float(percent)))
    img = img.resize((resized_width, resized_height), Image.ANTIALIAS)
    return img


def image_to_px(img):
    """
    Converts an image to an array stored as a DataFrame.

    :param img: An Image object.
    :return: The numeric representation of an image stored as a DataFrame.
    """
    px = pd.DataFrame(np.array(img).reshape(1, -1))
    return px


def eye_crop():
    """
    Crop the face just at "eye" position, and export these "eye" images as predictors.

    :return: a DataFrame including all predictors.
    """
    img_source = '../Datasets/cartoon_set/img/'
    img_suffix = '.png'
    save_path = '../Datasets/cartoon_set_b2/predictors_eye.csv'
    px = pd.DataFrame()
    for i in range(10000):
        img = Image.open(img_source + str(i) + img_suffix)
        img = img.crop((170, 240, 330, 290))
        img = resize_image(30, img)
        img_px = image_to_px(img)
        px = px.append(img_px, ignore_index=True)
    px.to_csv(save_path, index=False)
    return px


def detect_anomaly(predictors):
    """
    Detect outliers (wearing sunglasses).

    :param predictors: predictors in original dataset.
    :return: A list including all outliers' index.
    """
    anomaly_list = []
    for i in range(predictors.shape[0]):
        temp = 0
        for j in range(predictors.shape[1]):
            if ((int(predictors[i:i + 1][str(j)].values) < 40) == True):
                temp = temp + 1  # if the value of a feature is lower than 40, add 1.
        if (temp > 216):
            anomaly_list.append(
                i)  # if an instance has over 216 features with a value lower than 40, taken as an outlier.
    return anomaly_list


def remove_anomaly(anomaly_list, predictors, labels):
    """
    Remove outliers from the original dataset and generate a new dataset.

    :param anomaly_list: A list includes all outliers' index.
    :param predictors: All predictors from the original dataset.
    :param labels: All labels from the original dataset.
    :return: Predictors and labels after all anomalies are removed.
    """
    labels = pd.DataFrame(labels)
    predictors_normal = pd.DataFrame()
    labels_normal = pd.DataFrame()
    for i in range(predictors.shape[0]):
        if i not in anomaly_list:
            predictors_normal = predictors_normal.append(predictors[predictors.index == i], ignore_index=True)
            labels_normal = labels_normal.append(labels[labels.index == i], ignore_index=True)
    predictors_normal.to_csv('../Datasets/cartoon_set_b2/predictors_eye_normal.csv', index=False)
    labels_normal.to_csv('../Datasets/cartoon_set_b2/labels_eye_normal.csv', index=False)
    return predictors_normal, labels_normal


def load_data():
    """
    Load predictors and labels from .csv file.

    :return: All predictors and labels in a dataset.
    """
    predictors_path = '../Datasets/cartoon_set_b2/predictors_eye.csv'
    labels_path = '../Datasets/cartoon_set/labels.csv'
    predictors = pd.read_csv(predictors_path)
    labels = pd.read_csv(labels_path, delimiter='\t')['eye_color']
    return predictors, labels


def load_normal_data():
    """
    Load predictors and labels after the anomaly removal from .csv file.

    :return: All predictors and labels in a dataset.
    """
    predictors_path = 'Datasets/cartoon_set_b2/predictors_eye_normal.csv'
    labels_path = 'Datasets/cartoon_set_b2/labels_eye_normal.csv'
    predictors = pd.read_csv(predictors_path)
    labels = pd.read_csv(labels_path)['eye_color']
    return predictors, labels


def train_val_test_split(predictors, labels, test_ratio, val_ratio):
    """
    Split the dataset as training set, validation set and test set in given ratio.

    :param predictors: All predictors in a dataset.
    :param labels: All labels in a dataset.
    :param test_ratio: The ratio of test set to the whole dataset.
    :param val_ratio: The ratio of validation set to the training set.
    :return: The training set, validation set and test set.
    """
    np.random.seed(42)

    shuffled = np.random.permutation(len(predictors))  # add randomness.
    test_size = int(len(predictors) * test_ratio)
    X_temp = predictors.iloc[shuffled[test_size:]]
    X_test = predictors.iloc[shuffled[:test_size]]
    y_temp = labels.iloc[shuffled[test_size:]]
    y_test = labels.iloc[shuffled[:test_size]]

    shuffled = np.random.permutation(len(X_temp))  # add randomness.
    val_size = int(len(X_temp) * val_ratio)
    X_train = X_temp.iloc[shuffled[val_size:]]
    X_val = X_temp.iloc[shuffled[:val_size]]
    y_train = y_temp.iloc[shuffled[val_size:]]
    y_val = y_temp.iloc[shuffled[:val_size]]

    return [X_train, y_train], [X_val, y_val], [X_test, y_test]


def create_cvset(X_train, y_train, X_val, y_val, folds, i):
    """
    Rearrange new training set and validation set for each cross-validation.

    :param X_train: Predictors in training set.
    :param y_train: Labels in training set.
    :param X_val: Predictors in validation set.
    :param y_val: Labels in validation set.
    :param folds: The number of slice the training set split.
    :param i: The ith slice.
    :return: The rearranged training set and validation set.
    """
    y_train = np.array(y_train).reshape(-1, 2)  # reshape to two columns for vsplit.
    y_val = np.array(y_val).reshape(-1, 2)
    X_folds = np.vsplit(X_train, folds)
    y_folds = np.vsplit(y_train, folds)

    temp_X = X_val
    temp_y = y_val
    X_val = X_folds[i]
    y_val = y_folds[i]
    X_folds[i] = temp_X
    y_folds[i] = temp_y
    X_train = pd.DataFrame(np.vstack(X_folds))
    y_train = np.vstack(y_folds).reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)

    return X_train, y_train, X_val, y_val
