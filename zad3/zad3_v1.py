import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

from plotka import plot_decision_regions


class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return self.activation(self.net_input(X))
        # return np.where(self.net_input(X) >= 0.0, 1, 0)


class Classifier:
    def __init__(self, ppn1, ppn2):
        self.ppn1 = ppn1
        self.ppn2 = ppn2

    def predict(self, x):
        y = np.where(self.ppn2.predict(x) == 1, 2, 1)
        return np.where(self.ppn1.predict(x) == 1, 0, y)

def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    X_train_01_subset, y_train_01_subset, y_train_03_subset = prepare_class_subsets(X_train, y_train)

    print('01_subset ', y_train_01_subset)
    print('03_subset ', y_train_03_subset)

    ppn1 = LogisticRegressionGD(eta=0., n_iter=15000)
    ppn1.fit(X_train_01_subset, y_train_01_subset)

    ppn2 = LogisticRegressionGD(eta=0.55, n_iter=15000)
    ppn2.fit(X_train_01_subset, y_train_03_subset)

    calc_accuracy_total(X_train, ppn1, ppn2, y_train, y_train_01_subset, y_train_03_subset)

    # w perceptronie wyjście jest albo 1 albo -1
    # y_train_01_subset[(y_train_01_subset == 0)] = -1

    clas = Classifier(ppn1, ppn2)

    plot_decision_regions(X_train, y_train, classifier=clas)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


def prepare_class_subsets(X_train, y_train):
    y_train_01_subset = y_train.copy()
    y_train_03_subset = y_train.copy()
    X_train_01_subset = X_train.copy()

    y_train_01_subset[(y_train == 1) | (y_train == 2)] = 0
    y_train_01_subset[(y_train == 0)] = 1

    y_train_03_subset[(y_train == 0) | (y_train == 1)] = 0
    y_train_03_subset[(y_train == 2)] = 1

    # X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
    # y_train_01_subset = y_train_01_subset[(y_train == 0) | (y_train == 1)]
    # y_train_03_subset = y_train_03_subset[(y_train == 0) | (y_train == 1)]

    return X_train_01_subset, y_train_01_subset, y_train_03_subset


def calc_accuracy_total(X_train, ppn1, ppn2, y_train, y_train_01_subset, y_train_03_subset):
    y1_predict = ppn1.predict(X_train)
    y3_predict = ppn2.predict(X_train)
    accuracy_1 = accuracy(ppn1.predict(X_train), y_train_01_subset)
    accuracy_3 = accuracy(ppn2.predict(X_train), y_train_03_subset)
    print("accuracy1", accuracy_1)
    print("accuracy2", accuracy_3)

    if accuracy_1 > accuracy_3:
        y_results = np.where(y1_predict == 0, 0, np.where(y3_predict == 1, 2, 1))
    else:
        y_results = np.where(y3_predict == 0, 2, np.where(y1_predict == 1, 0, 1))
    print("acc_total", accuracy(y_results, y_train))


def accuracy(y_results, y_train):
    return (1 - np.mean(y_results != y_train)) * 100


if __name__ == '__main__':
    main()
