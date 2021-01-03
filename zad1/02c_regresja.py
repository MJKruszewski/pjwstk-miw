import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# a = np.loadtxt('Sharp_char.txt')
a = np.loadtxt('C:\\Users\\Maciej Kruszewski\\PycharmProjects\\miw\\zad1\\Sharp_char.txt')

x = a[:, [1]]
y = a[:, [0]]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)

c = np.hstack([x * x, x, np.ones(x.shape)])  # model y = ax2 + bx +c powinnismy użyć danych train X_train, y_train
v = np.linalg.pinv(c) @ y
e = sum((y - (v[0] * x * x + v[1] * x + v[2]))) ** 2/ len(x) #wektor błędu

c4 = np.hstack([X_train * X_train, X_train, np.ones(X_train.shape)])  # model y = ax2 + bx +c powinnismy użyć danych train X_train, y_train
v4 = np.linalg.pinv(c4) @ y_train



print(e)
# bład train
# bład test


c1 = np.hstack([1 / x, np.ones(x.shape)])
v1 = np.linalg.pinv(c1) @ y

plt.plot(x, y, 'g^')
plt.plot(x,  x * x + x + 1)  # kwadratowy
plt.plot(x, v[0] * x * x + v[1] * x + v[2])  # kwadratowy
plt.plot(x, v4[0] * x * x + v4[1] * x + v4[2])  # kwadratowy
# plt.plot(x, v1[0] / x + v1[1]) # odwrócony logarytmiczny


# plt.plot(x, y, 'g^')
# plt.plot(X_train, y_train, 'ro')






# stare

#
# c2 = np.hstack([x, np.ones(x.shape)])  # model y = ax +b
# v2 = np.linalg.pinv(c2) @ y
# e2 = sum((y - (v2[0] * x + v2[1])) ** 2)
#
# print(e2)
#
# plt.plot(x, y, 'ro')
# plt.plot(x, v[0] * x * x + v[1] * x + v[2])
# plt.plot(x, v2[0] * x + v2[1])
# plt.plot(x, v1[0] / x + v1[1])
plt.show()
