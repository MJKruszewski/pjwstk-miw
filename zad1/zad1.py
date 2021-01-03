import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import e as euler

# a = np.loadtxt('C:\\Users\\Maciej Kruszewski\\PycharmProjects\\miw\\zad1\\dane6.txt')
a = np.loadtxt('dane6.txt')

x = a[:, [0]]
y = a[:, [1]]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# liniowy
c2 = np.hstack([X_train, np.ones(X_train.shape)])  # model y = ax + 1
v2 = np.linalg.pinv(c2) @ y_train

e1 = sum((y - (v2[0] * x + v2[1]))) ** 2 / len(x)  #wektor błędu
print(e1)

plt.plot(x, y, 'g^')
plt.scatter(X_test, y_test,  color='black')
plt.plot(x, v2[0] * x + v2[1])
# ----

# Inny model
c = np.hstack([pow(X_train, 4), -3 * pow(X_train, 2), np.ones(X_train.shape)])  # model y = x^4 - 3*x^2 + 1
v = np.linalg.pinv(c) @ y_train # wektor parametró
model_train = v[0] * pow(x, 4) + (- 3) * v[1] * pow(x, 2) + 1
e = sum((y - model_train)) ** 2 / len(x)  #wektor błędu
print(e)

plt.plot(x, model_train, linewidth=3)
# ----


# Inny model - najlepszy
c_train = np.hstack(
    [
        0.0212 * X_train * X_train,
        1.7386 * X_train,
        np.ones(X_train.shape) * 33.409
    ]
)  # model y = 0,0212x2 - 1,7386x + 33,409
v_train = np.linalg.pinv(c_train) @ y_train # wektor parametrów

c_test = np.hstack(
    [
        0.0212 * X_test * X_test,
        1.7386 * X_test,
        np.ones(X_test.shape) * 33.409
    ]
)  # model y = 0,0212x2 - 1,7386x + 33,409
v_test = np.linalg.pinv(c_test) @ y_test # wektor parametrów

model_train = 0.0212 * v_train[0] * x * x - 1.7386 * v_train[1] * x + v_train[2] * 33.409
model_test = 0.0212 * v_test[0] * x * x - 1.7386 * v_test[1] * x + v_test[2] * 33.409

e_train = sum((y_train - (0.0212 * v_train[0] * X_train * X_train - 1.7386 * v_train[1] * X_train + v_train[2] * 33.409))) ** 2 / len(x)  #wektor błędu
e_test = sum((y_test - (0.0212 * v_train[0] * X_test * X_test - 1.7386 * v_train[1] * X_test + v_train[2] * 33.409))) ** 2 / len(x)  #wektor błędu
e_train_model = sum((y - model_train)) ** 2 / len(x)  #wektor błędu
e_test_model = sum((y - model_test)) ** 2 / len(x)  #wektor błędu

print(e_train_model)
print(e_test_model)
print(e_train)
print(e_test)
print(e1 > e_train_model)  # e2 jest lepszy niż e1 więc model 2 jest modelem dokładniejszym

plt.plot(x, model_train, linewidth=3)
# ----

plt.show()
