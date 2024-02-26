import numpy as np
import argparse

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_newton(X, y, learning_rate=0.001, n_iterations=1000, tol=1e-6):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(n_iterations):
        linear_model = np.dot(X, weights) + bias
        y_predicted = sigmoid(linear_model)

        # Compute gradient of the log-likelihood
        diff = y_predicted - y
        gradient = np.dot(X.T, diff) / n_samples

        # Compute Hessian matrix of the log-likelihood
        diag = y_predicted * (1 - y_predicted)
        H = np.dot(X.T, X * diag[:, np.newaxis]) / n_samples

        # Check for convergence (optional, not implemented here for simplicity)
        # if np.linalg.norm(gradient) < tol:
        #     break

        # Update parameters using Newton's method
        # Here we use np.linalg.pinv for pseudo-inverse to handle non-invertibility
        weights_update = np.linalg.pinv(H).dot(gradient)
        weights -= learning_rate * weights_update

        # Bias update is not strictly part of Newton's method but can be included for completeness
        # bias -= learning_rate * np.mean(diff)

    linear_model = np.dot(X, weights) + bias
    y_predicted = sigmoid(linear_model)
    y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
    return np.array(y_predicted_cls)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistic Regression with Newton's Method")
    parser.add_argument("--data", type=str, help="Path to data file (CSV format)")
    args = parser.parse_args()

    if args.data:
        data = np.genfromtxt(args.data, delimiter=',')
        X = data[:, :-1]
        y = data[:, -1]

        predictions = logistic_regression_newton(X, y)
        print("Predictions:", predictions)
    else:
        print("Please provide the path to the data file using the '--data' argument.")
