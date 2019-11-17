from B1 import data_preprocess, model_tuning


class B1:
    best_model = None

    def train(self, data_train, data_val):
        """
        :param data_train: Predictors and labels in training set.
        :param data_val: Predictors and labels in validation set.
        :return: The accuracy on training set.
        """
        best_model = model_tuning.tune_model(data_train[0], data_train[1], data_val[0], data_val[1])
        acc_train = model_tuning.measure_acc_train(best_model, data_train[0], data_train[1])
        self.best_model = best_model
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
    data_train, data_val, data_test = data_preprocess.train_val_test_split(predictors, labels, 0.1, 0.2)
    return data_train, data_val, data_test
