import numpy as np
from numpy.random import RandomState
from sklearn.metrics import mean_squared_error


class linearReg:
    epochs = 0
    learning_rate = None
    random_state = None
    coefficients = None

    def __init__(self, epochs=10000, learning_rate=0.0001, random_state=1):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

    def hypothesis_value(self, row):
        y_value = self.coefficients[0]
        for i in range(len(row)):
            y_value = y_value + self.coefficients[i + 1] * row[i]
        return y_value

    def fit(self, X, y):
        self.coefficients = np.asarray(RandomState(self.random_state).random_sample(len(X[0, :]) + 1))
        for i in range(self.epochs):
            for row in range(len(X[:, 0])):
                error = self.hypothesis_value(X[row, :]) - y[row]
                self.coefficients[0] = self.coefficients[0] - self.learning_rate * error[0]
                for j in range(1, len(self.coefficients)):
                    self.coefficients[j] = self.coefficients[j] - self.learning_rate * error[0] * X[row, j - 1]
            y_predict = [0.0 for i in range(len(X))]
            for row in range(len(X)):
                y_predict[row] = self.hypothesis_value(X[row, :])
            print("Error: %d" % mean_squared_error(y, y_predict))

    def predict(self, X, y):
        y_predict = np.array([0.0 for i in range(len(y))])
        for i in range(len(y)):
            y_predict[i] = self.hypothesis_value(X[i])
        return y_predict
