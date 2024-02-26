import numpy as np
import pandas as pd
import sys
import os


def linear_regression(data):
    """
    data: input data matrix with the last column being the target label
    return: rmse value
    """
    # Assuming the last feature in the list is the target label
    features = sys.argv[2:]
    input_features = features[:-1]
    target = features[-1]

    X = data[input_features].values
    y = data[target].values.reshape(-1, 1)

    ones = np.ones((X.shape[0], 1))
    X_b = np.hstack((ones, X))

    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    y_pred = X_b.dot(theta)

    rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    return rmse


def load_data():
    filename = sys.argv[1]
    feature_matrix = pd.read_csv(filename)
    feature_matrix = feature_matrix.dropna()
    features = sys.argv[2:]
    return feature_matrix[features]


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python linear_regression.py <path_to_csv> <input_feature1> <input_feature2> ... <target_feature>")
    else:
        data = load_data()
        RMSE_SCORE = linear_regression(data)
        print("RMSE score is : ", RMSE_SCORE)
