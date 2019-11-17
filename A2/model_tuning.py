from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import accuracy_score
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

# build a Convolutional Neural Networks with 4 conv. and 2 FC layers.
def build_model():
    """
    :return: An untrained CNN model.
    """
    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation='relu', input_shape=(73, 60, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(strides=(2, 2), pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(strides=(2, 2), pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=["accuracy"])
    return model


# train the CNN model.
def train_model(model, data_train, data_val):
    """
    :param model: An untrained CNN model.
    :param data_train: Predictors and labels in training set.
    :param data_val: Predictors and labels in validation set.
    :return: Training history and a fine-tuned model.
    """
    hist = model.fit(x=data_train[0],
                     y=data_train[1],
                     epochs=20,
                     validation_data=(data_val[0], data_val[1]),
                     batch_size=100,
                     callbacks=[LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)])
    return hist, model


# Measure the accuracy on training set.
def measure_acc_train(model, X_train, y_train):
    """
    :param model: The model with fine-tuned parameters that has been fit on data.
    :param X_train: The predictors in training set.
    :param y_train: The labels in training set.
    :return: The accuracy on training set.
    """
    y_temp = model.predict(X_train)
    y_pred = np.argmax(y_temp, axis=1)
    acc_train = accuracy_score(y_train, y_pred)
    return round(acc_train, 2)


# Measure the accuracy on test set.
def measure_acc_test(model, X_test, y_test):
    """
    :param model: The model with fine-tuned parameters and has been fit on training set.
    :param X_test: The predictors in test set.
    :param y_test: The labels in test set.
    :return: The accuracy on test set.
    """
    y_temp = model.predict(X_test)
    y_pred = np.argmax(y_temp, axis=1)
    acc_test = accuracy_score(y_test, y_pred)
    return round(acc_test, 2)


# Load a pre-trained CNN model.
def load_trained():
    """
    :return: A pre-trained model.
    """
    model = load_model('A2/pre-trained_model.h5')
    return model
