import pandas as pd
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
from keras.utils.np_utils import to_categorical


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


def shrink_image():
    """
    Shrink images to the shape of 73*60*3.

    :return: predictors after being shrunk.
    """
    img_source = '../Datasets/celeba/img/'
    img_suffix = '.jpg'
    save_path = '../Datasets/celeba_set_a2/predictors_shrink.csv'
    px = pd.DataFrame()
    for i in range(5000):
        img = Image.open(img_source + str(i) + img_suffix)
        img = resize_image(60, img)
        img_px = image_to_px(img)
        px = px.append(img_px, ignore_index=True)
    px.to_csv(save_path, index=False)
    return px


def data_augmentation(img):
    """
    Data augmentation via flipping an image horizontally.

    :param img: An Image object.
    :return: A new image represented by a ndarray.
    """
    aug = iaa.Sequential([
        iaa.Fliplr(1.0),
        # iaa.Crop(percent=(0, 0.2),
        # iaa.GaussianBlur(1),
        # iaa.Crop(percent=(0, 0.2),
        # iaa.Affine(rotate=(-10, 10))
    ])
    img_aug = aug.augment_image(img)
    return img_aug


def aug_trainset(X_train):
    """
    Export factitious data as .csv file.

    :param X_train: Predictors of training set.
    :return: factitious predictors.
    """
    save_path = '../Datasets/celeba_set_a2/X_train_flip.csv'
    X_train_aug = pd.DataFrame()
    X_train = np.array(X_train).reshape(-1, 73, 60, 3)
    for i in range(X_train.shape[0]):
        aug = data_augmentation(X_train[i])
        aug = pd.DataFrame(aug.reshape(1, -1))
        X_train_aug = X_train_aug.append(aug, ignore_index=True)
    X_train_aug.to_csv(save_path, index=False)
    return X_train_aug


def load_data():
    """
    Load predictors and labels from .csv file.

    :return: All predictors and labels in a dataset.
    """
    predictors_path = 'Datasets/celeba_set_a2/predictors_shrink.csv'
    labels_path = 'Datasets/celeba/labels.csv'
    predictors = pd.read_csv(predictors_path)
    labels = pd.read_csv(labels_path, delimiter='\t')['smiling']
    return predictors, labels


def load_train_aug():
    """
    Load factitious predictors and labels from .csv file.

    :return: factitious predictors.
    """
    predictors_path = 'Datasets/celeba_set_a2/X_train_flip.csv'
    predictors = pd.read_csv(predictors_path)
    return predictors


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


def data_prepare(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Prepare data before fitting a CNN model.

    :param data_train: Predictors and labels in training set.
    :param data_val: Predictors and labels in validation set.
    :param data_test: Predictors and labels in test set.
    :return: The training set, validation set and test set.
    """
    X_train = np.array(X_train).reshape(-1, 73, 60, 3)
    X_train = X_train.astype("float32") / 255.

    X_val = np.array(X_val).reshape(-1, 73, 60, 3)
    X_val = X_val.astype("float32") / 255.

    X_test = np.array(X_test).reshape(-1, 73, 60, 3)
    X_test = X_test.astype("float32") / 255.

    y_train = pd.DataFrame(y_train)
    y_train.smiling[y_train['smiling'] == -1] = 0
    y_temp = y_train
    y_train = to_categorical(y_train)

    y_val = pd.DataFrame(y_val)
    y_val.smiling[y_val['smiling'] == -1] = 0
    y_val = to_categorical(y_val)

    y_test = pd.DataFrame(y_test)
    y_test.smiling[y_test['smiling'] == -1] = 0

    return [X_train, y_train, y_temp], [X_val, y_val], [X_test, y_test]
