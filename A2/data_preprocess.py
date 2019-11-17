import pandas as pd
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
from keras.utils.np_utils import to_categorical


# Resize the width of an image and its height in proportion.
def resize_image(resized_width, img):
    """
    :param resized_width: The width of an image after being resized.
    :param img: An Image object.
    :return: The Image object resized.
    """
    percent = (resized_width / float(img.size[0]))
    resized_height = int((float(img.size[1]) * float(percent)))
    img = img.resize((resized_width, resized_height), Image.ANTIALIAS)
    return img


# Converts an image to an array stored as a DataFrame.
def image_to_px(img):
    """
    :param img: An Image object.
    :return: The numeric representation of an image stored as a DataFrame.
    """
    px = pd.DataFrame(np.array(img).reshape(1, -1))
    return px


# Shrink images to the shape of 73*60*3.
def shrink_image():
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
    return 0


# Data augmentation via flipping an image horizontally.
def data_augmentation_flip(img):
    """
    :param img: An Image object.
    :return: A new image represented by a ndarray.
    """
    aug = iaa.Sequential([
        iaa.Fliplr(1.0),
        # iaa.Crop(percent=(0, 0.2)
    ])
    img_aug = aug.augment_image(np.array(img))
    return img_aug


# Export factitious data as .csv file.
def flip_image():
    img_source = '../Datasets/celeba/img/'
    img_suffix = '.jpg'
    save_path = '../Datasets/celeba_set_a2/predictors_flip.csv'
    px = pd.DataFrame()
    for i in range(5000):
        img = Image.open(img_source + str(i) + img_suffix)
        img = resize_image(60, img)
        img_aug = data_augmentation_flip(img)
        img_px = image_to_px(img_aug)
        px = px.append(img_px, ignore_index=True)
    px.to_csv(save_path, index=False)
    return 0


# Export factitious data as .csv file.
def crop_image():
    img_source = '../Datasets/celeba/img/'
    img_suffix = '.jpg'
    save_path = '../Datasets/celeba_set_a2/predictors_crop.csv'
    px = pd.DataFrame()
    for i in range(5000):
        img = Image.open(img_source + str(i) + img_suffix)
        img = img.crop((50, 80, 150, 180))
        img_aug = img.resize((60, 73))
        img_px = image_to_px(img_aug)
        px = px.append(img_px, ignore_index=True)
    px.to_csv(save_path, index=False)
    return 0


# Load predictors and labels from .csv file.
def load_data():
    """
    :return: All predictors and labels in a dataset.
    """
    predictors_path = 'Datasets/celeba_set_a2/predictors_shrink.csv'
    labels_path = 'Datasets/celeba/labels.csv'
    predictors = pd.read_csv(predictors_path)
    labels = pd.read_csv(labels_path, delimiter='\t')['smiling']
    return predictors, labels


# Load factitious predictors and labels from .csv file.
def load_aug_flip():
    """
    :return: All predictors and labels in a dataset.
    """
    predictors_path = 'Datasets/celeba_set_a2/predictors_flip.csv'
    labels_path = 'Datasets/celeba/labels.csv'
    predictors = pd.read_csv(predictors_path)
    labels = pd.read_csv(labels_path, delimiter='\t')['smiling']
    return predictors, labels


# Load factitious predictors and labels from .csv file.
def load_aug_crop():
    """
    :return: All predictors and labels in a dataset.
    """
    predictors_path = 'Datasets/celeba_set_a2/predictors_crop.csv'
    labels_path = 'Datasets/celeba/labels.csv'
    predictors = pd.read_csv(predictors_path)
    labels = pd.read_csv(labels_path, delimiter='\t')['smiling']
    return predictors, labels


# Split the dataset as training set, validation set and test set in given ratio.
def train_val_test_split(predictors, labels, test_ratio, val_ratio):
    """
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

    X_train_flip, y_train_flip = load_aug_flip()
    X_train = X_train.append(X_train_flip, ignore_index=True)
    y_train = y_train.append(y_train_flip, ignore_index=True)
    X_train_crop, y_train_crop = load_aug_crop()
    X_train = X_train.append(X_train_crop, ignore_index=True)
    y_train = y_train.append(y_train_crop, ignore_index=True)

    return [X_train, y_train], [X_val, y_val], [X_test, y_test]


# prepare data before fitting a CNN model.
def data_prepare(X_train, y_train, X_val, y_val, X_test, y_test):
    """
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
