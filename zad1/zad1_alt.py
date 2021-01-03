import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
# a = np.loadtxt('Sharp_char.txt')
a = np.loadtxt('C:\\Users\\Maciej Kruszewski\\PycharmProjects\\miw\\zad1\\dane6.txt')

x = a[:, [1]]
y = a[:, [0]]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

Y_predicted = regr.predict(X_test)

print('Mean squared error: %.2f'
      % mean_squared_error(y_test, Y_predicted))

print('Coefficient of determination: %.2f'
      % r2_score(y_test, Y_predicted))

plt.plot(x, y, 'g^')
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, Y_predicted, color='blue', linewidth=3)
plt.show()
