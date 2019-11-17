from A2 import data_preprocess, model_tuning


class A2:
    best_model = None

    def train(self, data_train, data_val, load_model: bool):
        """
        :param data_train: Predictors and labels in training set.
        :param data_val: Predictors and labels in validation set.
        :param load_model: Set True to load pre-trained model, otherwise train the model from scratch.
        :return: The accuracy on training set.
        """
        acc_train = 0
        model = None
        if load_model:
            # print('loading')
            model = model_tuning.load_trained()
            acc_train = model_tuning.measure_acc_train(model, data_train[0], data_train[2])
        elif not load_model:
            model = model_tuning.build_model()
            # print('training')
            hist, model = model_tuning.train_model(model, data_train, data_val)
            # print(hist)
            acc_train = model_tuning.measure_acc_train(model, data_train[0], data_train[2])
        self.best_model = model
        return acc_train

    def test(self, data_test):
        """
        :param data_test: Predictors and labels in test set.
        :return: The accuracy on test set.
        """
        acc_test = model_tuning.measure_acc_test(self.best_model, data_test[0], data_test[1])
        return acc_test


def data_preprocessing():
    """
    :return: The training set, validation set and test set.
    """
    predictors, labels = data_preprocess.load_data()
    data_train, data_val, data_test = data_preprocess.train_val_test_split(predictors, labels, 0.2, 0.25)
    data_train, data_val, data_test = data_preprocess.data_prepare(data_train[0], data_train[1], data_val[0],
                                                                   data_val[1], data_test[0], data_test[1])
    return data_train, data_val, data_test
