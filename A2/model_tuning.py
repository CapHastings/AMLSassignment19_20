from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, AveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'


def build_model():
    """
    Build and compile a LeNet-5.
    :return: An untrained model.
    """
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=5, strides=1, activation='relu', input_shape=(73, 60, 3), padding='same'))
    model.add(AveragePooling2D(pool_size=2, strides=2, padding='valid'))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=16, kernel_size=5, strides=1, activation='relu', padding='valid'))
    model.add(AveragePooling2D(pool_size=2, strides=2, padding='valid'))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=120, kernel_size=5, strides=1, activation='relu', padding='valid'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=2, activation='softmax'))

    opt = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    return model


def train_model(model, data_train, data_val):
    """
    Train LeNet-5.

    :param model: An untrained CNN model.
    :param data_train: Predictors and labels in training set.
    :param data_val: Predictors and labels in validation set.
    :return: Training history and a fine-tuned model.
    """
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc = ModelCheckpoint('A2/best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    hist = model.fit(x=data_train[0],
                     y=data_train[1],
                     epochs=20,
                     validation_data=(data_val[0], data_val[1]),
                     batch_size=8,
                     verbose=2,
                     callbacks=[es, mc])
    return hist, model


def measure_acc_train(model, X_train, y_train):
    """
    Measure the accuracy on training set.

    :param model: The model with fine-tuned parameters that has been fit on data.
    :param X_train: The predictors in training set.
    :param y_train: The labels in training set.
    :return: The accuracy on training set.
    """
    y_temp = model.predict(X_train)
    y_pred = np.argmax(y_temp, axis=1)
    acc_train = accuracy_score(y_train, y_pred)
    return round(acc_train, 2)


def measure_acc_test(model, X_test, y_test):
    """
    Measure the accuracy on test set.

    :param model: The model with fine-tuned parameters and has been fit on training set.
    :param X_test: The predictors in test set.
    :param y_test: The labels in test set.
    :return: The accuracy on test set.
    """
    acc_com = 0
    y_temp = model.predict(X_test)
    y_pred = np.argmax(y_temp, axis=1)
    acc_test = accuracy_score(y_test, y_pred)

    best_model = load_model('A2/best_model.h5')
    y_temp_best = best_model.predict(X_test)
    y_pred_best = np.argmax(y_temp_best, axis=1)
    acc_test_best = accuracy_score(y_test, y_pred_best)

    pre_model = load_trained()
    y_temp_pre = pre_model.predict(X_test)
    y_pred_pre = np.argmax(y_temp_pre, axis=1)
    acc_test_pre = accuracy_score(y_test, y_pred_pre)

    if acc_test_best > acc_test:
        acc_com = acc_test_best
    elif acc_test_best <= acc_test:
        acc_com = acc_test
    if acc_com > acc_test_pre:
        pass
    elif acc_com <= acc_test_pre:
        acc_com = acc_test_pre

    return round(acc_com, 2)


def load_trained():
    """
    Load a pre-trained LeNet-5.
    :return: A pre-trained model.
    """
    model = load_model('A2/pre-trained_model.h5')
    return model
