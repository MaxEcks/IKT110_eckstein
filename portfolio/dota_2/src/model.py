import numpy as np
from numba import njit

@njit
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


@njit
def predict(theta, xs, batch):
    bias = theta[0]
    weight = theta[1:]
    linear = bias + np.dot(xs[batch], weight)
    return sigmoid(linear)                      # logistic regression output

@njit
def predict_single_value(theta, input):
    bias = theta[0]
    weight = theta[1:]
    linear = bias + np.dot(input, weight)
    return sigmoid(linear)

@njit
def J_logistic(theta, xs, y, batch):
    h = predict(theta, xs, batch)               # probability
    y_batch = y[batch]

    # numerically stable logistic loss
    eps = 1e-12
    loss = - (y_batch * np.log(h + eps) + (1 - y_batch) * np.log(1 - h + eps))
    return loss.sum()

@njit
def gradient_J_logistic(theta, xs, y, batch):
    h = predict(theta, xs, batch)
    y_batch = y[batch]
    error = h - y_batch                          # logistic error

    # Gradient wrt bias
    dp = error.sum()

    # Gradient wrt weights
    dw = np.dot(xs[batch].T, error)

    grad = np.empty(theta.shape)
    grad[0] = dp
    grad[1:] = dw
    return grad