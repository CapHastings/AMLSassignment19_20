from sklearn.metrics import accuracy_score
from a1 import A1, data_preprocessing
import matplotlib.pyplot as plt

train_acc, val_acc = [], []


def plot_learning_curves(model, X_train, y_train, X_val, y_val):
    for m in range(100, len(X_train), 100):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_acc.append(accuracy_score(y_train_predict, y_train[:m]))
        val_acc.append(accuracy_score(y_val_predict, y_val))


best_model = A1().best_model
data_train, data_val, data_test = data_preprocessing()
X_train = data_train[0]
y_train = data_train[1]
X_val = data_val[0]
y_val = data_val[1]

plot_learning_curves(best_model, X_train, y_train, X_val, y_val)
plt.plot(train_acc, "r", linewidth=2.5, label="Training")
plt.plot(val_acc, "b", linewidth=2.5, label="Validation")
plt.legend(loc="best")
plt.xlabel("Training set size")
plt.ylabel("Accuracy")
plt.show()
