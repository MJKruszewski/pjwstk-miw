import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1 - warstwa wejścia
# 1000 - Warstwa ukryta/1000 neuronów na niej
# 1 - Warstwa wyjścia
globalLayers = [1, 1000, 1]
globalActivations = ['tanh', 'relu']
globalWeights = []
globalBiases = []

# Wypełniamy tablice wag i biasów losowymi wartościami, które w późniejszym etapie będą nadpisywane
for i in range(len(globalLayers) - 1):
    globalWeights.append(np.random.randn(globalLayers[i + 1], globalLayers[i]))
    globalBiases.append(np.random.randn(globalLayers[i + 1], 1))


def feedforward(x):
    # Przygotowanie wyniku wyjścia dla sigmoidalnej/relu funkcji aktywacji
    a = np.copy(x)
    z_s = []
    a_s = [a]
    for i in range(len(globalWeights)):
        activation_function = get_activation_function(globalActivations[i])
        z_s.append(globalWeights[i].dot(a) + globalBiases[i])
        a = activation_function(z_s[-1])
        a_s.append(a)
    return z_s, a_s


def calculate(x):
    _, a_s = feedforward(x)

    return a_s


def backpropagation(y, z_s, a_s):
    dw = []  # dC/dW
    db = []  # dC/dB
    deltas = [None] * len(globalWeights)  # delta = dC/dZ  błąd dla każdej warstwy
    # wstawiamy błąd z ostatniej warstwy
    deltas[-1] = ((y - a_s[-1]) * (get_derivative_activation_function(globalActivations[-1]))(z_s[-1]))

    # Tutaj rozpoczynamy magię backpropgation
    for i in reversed(range(len(deltas) - 1)):
        deltas[i] = globalWeights[i + 1].T.dot(deltas[i + 1]) * (get_derivative_activation_function(globalActivations[i])(z_s[i]))
        db = [d.dot(np.ones((y.shape[1], 1))) / float(y.shape[1]) for d in deltas]
        dw = [d.dot(a_s[i].T) / float(y.shape[1]) for i, d in enumerate(deltas)]

    # zwracamy pochodne w odniesieniu do macierzy wag i odchyleń
    return dw, db


def train(x, y, epochs, lr):
    """
    :param x: wejście
    :param y: wyjście
    :param epochs: liczba epok (iteracji uczenia)
    :param lr: (wspólczynnik definiujący spadek szybkość uczenia się sieci)
    :return:
    """

    global globalWeights
    global globalBiases
    # zaktualizuj wagi i odchylenia (wzorce) na podstawie danych wyjściowych | uczenie sieci metodą online
    for e in range(epochs):
        z_s, a_s = feedforward(x)
        dw, db = backpropagation(y, z_s, a_s)
        globalWeights = [w + lr * dweight for w, dweight in zip(globalWeights, dw)]
        globalBiases = [w + lr * dbias for w, dbias in zip(globalBiases, db)]

        print("strata -> {}".format(np.linalg.norm(a_s[-1] - y)))


def get_activation_function(name):
    # Funkcja aktywacji

    if name == 'sigmoid':
        return lambda x: np.exp(x) / (1 + np.exp(x))
    elif name == 'tanh':
        return lambda x: np.sinh(x) / np.cosh(x)
    elif name == 'relu':
        def relu(x):
            y = np.copy(x)
            y[y < 0] = 0
            return y

        return relu


def get_derivative_activation_function(name):
    # Funkcja aktywacji pochodnej

    if name == 'sigmoid':
        sig = lambda x: np.exp(x) / (1 + np.exp(x))
        return lambda x: sig(x) * (1 - sig(x))
    elif name == 'tanh':
        tanh = lambda x: np.sinh(x) / np.cosh(x)
        return lambda x: tanh(x) * (1 - tanh(x))
    elif name == 'relu':
        def relu_diff(x):
            y1 = np.copy(x)
            y1[y1 >= 0] = 1
            y1[y1 < 0] = 0
            return y1

        return relu_diff


a = np.loadtxt('dane6.txt')

X = a[:, [0]]
y = a[:, [1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X = X.transpose()
y = y.transpose()

X_train = X_train.transpose()
y_train = y_train.transpose()
X_test = X_test.transpose()
y_test = y_test.transpose()

train(X_train, y_train, epochs=10000, lr=0.001)
train_results = calculate(X_train)
test_results = calculate(X_test)
# print(y, X)
plt.scatter(X, y, color='green')
plt.scatter(X_test, y_test, color='yellow')

plt.scatter(X_train.flatten(), train_results[-1].flatten(), color='black', linewidth=2)
plt.scatter(X_test.flatten(), test_results[-1].flatten(), color='red', linewidth=2)
plt.show()
