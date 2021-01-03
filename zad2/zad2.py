import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class NeuralNetwork(object):
    def __init__(self, layers=[2, 10, 1], activations=['sigmoid', 'sigmoid']):
        assert (len(layers) == len(activations) + 1)
        self.layers = [1, 1000, 1]
        self.activations = ['sigmoid', 'relu']
        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i + 1], layers[i]))
            self.biases.append(np.random.randn(layers[i + 1], 1))

    def feedforward(self, x):
        # return the feedforward value for x
        a = np.copy(x)
        z_s = []
        a_s = [a]
        for i in range(len(self.weights)):
            activation_function = self.getActivationFunction(self.activations[i])
            z_s.append(self.weights[i].dot(a) + self.biases[i])
            a = activation_function(z_s[-1])
            a_s.append(a)
        return (z_s, a_s)

    def backpropagation(self, y, z_s, a_s):
        dw = []  # dC/dW
        db = []  # dC/dB
        deltas = [None] * len(self.weights)  # delta = dC/dZ  known as error for each layer
        # insert the last layer error
        deltas[-1] = ((y - a_s[-1]) * (self.getDerivitiveActivationFunction(self.activations[-1]))(z_s[-1]))
        # Perform BackPropagation
        for i in reversed(range(len(deltas) - 1)):
            deltas[i] = self.weights[i + 1].T.dot(deltas[i + 1]) * (
                self.getDerivitiveActivationFunction(self.activations[i])(z_s[i]))
            # a= [print(d.shape) for d in deltas]
            batch_size = y.shape[1]
            db = [d.dot(np.ones((batch_size, 1))) / float(batch_size) for d in deltas]
            dw = [d.dot(a_s[i].T) / float(batch_size) for i, d in enumerate(deltas)]
            # return the derivitives respect to weight matrix and biases
            return dw, db

    def train(self, x, y, batch_size=10, epochs=100, lr=0.01):
        # update weights and biases based on the output
        for e in range(epochs):
            i = 0
            while (i < len(y)):
                x_batch = x[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                i = i + batch_size
                z_s, a_s = self.feedforward(x_batch)
                dw, db = self.backpropagation(y_batch, z_s, a_s)
                self.weights = [w + lr * dweight for w, dweight in zip(self.weights, dw)]
                self.biases = [w + lr * dbias for w, dbias in zip(self.biases, db)]
                print("loss = {}".format(np.linalg.norm(a_s[-1] - y_batch)))

    @staticmethod
    def getActivationFunction(name):
        if (name == 'sigmoid'):
            return lambda x: np.exp(x) / (1 + np.exp(x))
        elif (name == 'tanh'):
            return lambda x: np.sinh(x) / np.cosh(x)
        elif (name == 'linear'):
            return lambda x: x
        elif (name == 'relu'):
            def relu(x):
                y = np.copy(x)
                y[y < 0] = 0
                return y

            return relu
        else:
            print('Unknown activation function. linear is used')
            return lambda x: x

    @staticmethod
    def getDerivitiveActivationFunction(name):
        if (name == 'sigmoid'):
            sig = lambda x: np.exp(x) / (1 + np.exp(x))
            return lambda x: sig(x) * (1 - sig(x))
        elif (name == 'tanh'):
            tanh = lambda x: np.sinh(x) / np.cosh(x)
            return lambda x: tanh(x) * (1 - tanh(x))
        elif (name == 'linear'):
            return lambda x: 1
        elif (name == 'relu'):
            def relu_diff(x):
                y1 = np.copy(x)
                y1[y1 >= 0] = 1
                y1[y1 < 0] = 0
                return y1

            return relu_diff
        else:
            print('Unknown activation function. linear is used')
            return lambda x: 1


nn = NeuralNetwork([1, 1000, 1], activations=['sigmoid', 'relu'])
a = np.loadtxt('dane6.txt')

X = a[:, [0]]
y = a[:, [1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

X = X.transpose()
y = y.transpose()

X_train = X_train.transpose()
y_train = y_train.transpose()
X_test = X_test.transpose()
y_test = y_test.transpose()

nn.train(X_train, y_train, epochs=1000, batch_size=5000, lr=0.001)
_, a_s = nn.feedforward(X_train)
_, a_s2 = nn.feedforward(X_test)
# print(y, X)
plt.scatter(X, y, color='green')
plt.scatter(X_test, y_test, color='yellow')

plt.scatter(X_train.flatten(), a_s[-1].flatten(), color='black', linewidth=2)
plt.scatter(X_test.flatten(), a_s2[-1].flatten(), color='red', linewidth=2)
plt.show()
