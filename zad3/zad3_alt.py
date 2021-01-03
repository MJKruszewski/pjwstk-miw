import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

from plotka import plot_decision_regions


class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1+ X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update *xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


class Classifier:
    def __init__(self, ppn1, ppn2, ppn3):
        self.ppn1 = ppn1
        self.ppn2 = ppn2
        self.ppn3 = ppn3

    def predict(self, x):
        y1 = np.where(self.ppn3.predict(x) == 1, 3, 2)
        y = np.where(self.ppn2.predict(x) == 1, 1, y1)
        return np.where(self.ppn1.predict(x) == 1, 0, y)


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    y_train_01_subset = y_train.copy()
    y_train_02_subset = y_train.copy()
    y_train_03_subset = y_train.copy()
    X_train_01_subset = X_train.copy()

    y_train_01_subset[(y_train == 1) | (y_train == 2)] = -1
    y_train_01_subset[(y_train_01_subset == 0)] = 1

    y_train_02_subset[(y_train == 1) | (y_train == 1)] = -1
    y_train_02_subset[(y_train_02_subset == 1)] = 1

    y_train_03_subset[(y_train == 1) | (y_train == 0)] = -1
    y_train_03_subset[(y_train_03_subset == 2)] = 1

    print('y_train_01_subset ', y_train_01_subset)
    print('y_train_01_subset ', y_train_02_subset)
    print('y_train_03_subset ', y_train_03_subset)

    ppn1 = Perceptron(eta=0.1, n_iter=500)
    ppn1.fit(X_train_01_subset, y_train_01_subset)
    ppn2 = Perceptron(eta=0.1, n_iter=500)
    ppn2.fit(X_train_01_subset, y_train_03_subset)
    ppn3 = Perceptron(eta=0.1, n_iter=500)
    ppn3.fit(X_train_01_subset, y_train_02_subset)

    y1_predict = ppn1.predict(X_train)
    y3_predict = ppn2.predict(X_train)
    accuracy_1 = accuracy(ppn1.predict(X_train), y_train_01_subset)
    accuracy_3 = accuracy(ppn2.predict(X_train), y_train_03_subset)
    print("acc1", accuracy_1)
    print("acc2", accuracy_3)

    if accuracy_1 > accuracy_3:
        y_results = np.where(y1_predict == 0, 0, np.where(y3_predict == 1, 2, 1))
    else:
        y_results = np.where(y3_predict == 0, 2, np.where(y1_predict == 1, 0, 1))

    print("acc_total", accuracy(y_results, y_train))

    # w perceptronie wyjście jest albo 1 albo -1
    # y_train_01_subset[(y_train_01_subset == 0)] = -1

    clas = Classifier(ppn1, ppn2, ppn3)

    plot_decision_regions(X_train, y_train, classifier=clas)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()

    # plot_decision_regions(X_test, y_test, classifier=clas)
    # plt.xlabel(r'$x_1$')
    # plt.ylabel(r'$x_2$')
    # plt.legend(loc='upper left')
    # plt.show()


def accuracy(y_results, y_train):
    return (1 - np.mean(y_results != y_train)) * 100


if __name__ == '__main__':
    main()
