from PIL import Image
import pandas as pd
import numpy as np
from imgaug import augmenters as iaa
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


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

    :return: a DataFrame including all predictors
    """
    img_source = '../Datasets/celeba/img/'
    img_suffix = '.jpg'
    save_path = '../Datasets/celeba_set_a1/predictors_shrink.csv'
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
    :param X_train:
    :return:
    """
    save_path = '../Datasets/celeba_set_a1/X_train_flip.csv'
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
    predictors_path = '../Datasets/celeba_set_a1/predictors_shrink.csv'
    labels_path = '../Datasets/celeba/labels.csv'
    predictors = pd.read_csv(predictors_path)
    labels = pd.read_csv(labels_path, delimiter='\t')['gender']
    return predictors, labels


def load_train_aug():
    """
    Load factitious predictors and labels from .csv file.

    :return: factitious predictors.
    """
    predictors_path = '../Datasets/celeba_set_a1/X_train_flip.csv'
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
    y_folds = np.vsplit(y_train, folds)
    X_folds = np.vsplit(X_train, folds)

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


def pca_data(data_train, data_val, data_test):
    """
    Conduct dimensionality reduction via PCA, and export after-PCA data.

    :param data_train: Predictors and labels in training set.
    :param data_val: Predictors and labels in validation set.
    :param data_test: Predictors and labels in test set.
    :return: the training data, validation dara and test data after PCA.
    """
    data_train[0] = MinMaxScaler().fit_transform(data_train[0])
    data_val[0] = MinMaxScaler().fit_transform(data_val[0])
    data_test[0] = MinMaxScaler().fit_transform(data_test[0])
    pca = PCA(n_components=0.99)
    data_train[0] = pca.fit_transform(data_train[0])
    data_val[0] = pca.transform(data_val[0])
    data_test[0] = pca.transform(data_test[0])

    pd.DataFrame(data_train[0]).to_csv('../Datasets/celeba_set_a1/predictors_pca_train.csv', index=False)
    pd.DataFrame(data_val[0]).to_csv('../Datasets/celeba_set_a1/predictors_pca_val.csv', index=False)
    pd.DataFrame(data_test[0]).to_csv('../Datasets/celeba_set_a1/predictors_pca_test.csv', index=False)
    pd.DataFrame(data_train[1]).to_csv('../Datasets/celeba_set_a1/labels_pca_train.csv', index=False)
    pd.DataFrame(data_val[1]).to_csv('../Datasets/celeba_set_a1/labels_pca_val.csv', index=False)
    pd.DataFrame(data_test[1]).to_csv('../Datasets/celeba_set_a1/labels_pca_test.csv', index=False)
    return data_train, data_val, data_test


def load_pca_data():
    """
    Load after-PCA predictors and labels from .csv file.

    :return: The training set, validation set and test set based on given ratio.
    """
    X_train = pd.read_csv('Datasets/celeba_set_a1/predictors_pca_train.csv')
    y_train = pd.read_csv('Datasets/celeba_set_a1/labels_pca_train.csv')['gender']
    X_val = pd.read_csv('Datasets/celeba_set_a1/predictors_pca_val.csv')
    y_val = pd.read_csv('Datasets/celeba_set_a1/labels_pca_val.csv')['gender']
    X_test = pd.read_csv('Datasets/celeba_set_a1/predictors_pca_test.csv')
    y_test = pd.read_csv('Datasets/celeba_set_a1/labels_pca_test.csv')['gender']

    return [X_train, y_train], [X_val, y_val], [X_test, y_test]

# ======================================================================================================================
# It is not necessary to run following codes, since the prepared data have been exported.
# predictors,labels=load_data() # load data from the dataset.
# data_train,data_val,data_test=train_val_test_split(predictors,labels,0.2,0.2) # split the dataset as given ratio.

# data augmentation on training set
# X_train_aug=aug_trainset(data_train[0])
# data_train[0]=data_train[0].append(X_train_aug,ignore_index=True)
# y_train_aug=data_train[1]
# data_train[1]=data_train[1].append(y_train_aug,ignore_index=True)

# dimensionality reduction via PCA
# data_train, data_val, data_test=pca_data(data_train,data_val,data_test) # takes time, use load_pca_data() instead.
# ======================================================================================================================
