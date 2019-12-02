import matplotlib.pyplot as plt
from a2 import A2

hist = A2().hist


def plot_learning_curves_acc(hist):
    plt.plot(hist.history['accuracy'], color='b', linewidth=2.5, label="Training")
    plt.plot(hist.history['val_accuracy'], color='r', linewidth=2.5, label="Validation")
    plt.legend(loc="best")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()


def plot_learning_curves_loss(hist):
    plt.plot(hist.history['loss'], color='b', linewidth=2.5, label="Training")
    plt.plot(hist.history['val_loss'], color='r', linewidth=2.5, label="Validation")
    plt.legend(loc="best")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
