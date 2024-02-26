import numpy as np
import pandas as pd
import math
import sys
import os


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, weights):
    m = X.shape[0]
    y_pred = sigmoid(np.dot(X, weights))
    cost = (-1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return cost


def gradient_descent(X, y, weights, learning_rate, iterations):
    m = X.shape[0]
    for i in range(iterations):
        y_pred = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (y_pred - y)) / m
        weights -= learning_rate * gradient
    return weights


def predict(X, weights):
    y_pred_prob = sigmoid(np.dot(X, weights))
    y_pred = np.where(y_pred_prob >= 0.5, 1, 0)
    return y_pred


def logistic_regression(xtrain, ytrain, xtest, ytest):
    xtrain = np.hstack((np.ones((xtrain.shape[0], 1)), xtrain))  # Add bias term
    xtest = np.hstack((np.ones((xtest.shape[0], 1)), xtest))  # Add bias term

    weights = np.zeros(xtrain.shape[1])
    learning_rate = 0.01
    iterations = 1000

    weights = gradient_descent(xtrain, ytrain, weights, learning_rate, iterations)

    y_pred = predict(xtest, weights)
    accuracy = np.mean(y_pred == ytest)

    return accuracy



# do not modify this function
def load_data():
    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    train_feature_matrix = pd.read_csv(train_filename) 
    test_feature_matrix = pd.read_csv(test_filename)
    train_feature_matrix = train_feature_matrix.dropna()
    test_feature_matrix = test_feature_matrix.dropna()
    X_TRAIN = train_feature_matrix.iloc[:, :len(train_feature_matrix.columns)-1] 
    Y_TRAIN = train_feature_matrix.iloc[:, -1]
    X_TEST = test_feature_matrix.iloc[:, :len(test_feature_matrix.columns)-1] 
    Y_TEST = test_feature_matrix.iloc[:, -1]
    return X_TRAIN, Y_TRAIN, X_TEST, Y_TEST


if __name__ == "__main__":
    xtrain, ytrain, xtest, ytest = load_data()
    ACCURACY_SCORE = logistic_regression(xtrain, ytrain, xtest, ytest)
    print("ACCURACY score is : ", ACCURACY_SCORE)
