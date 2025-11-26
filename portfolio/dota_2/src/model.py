import numpy as np
from numba import njit


@njit
def sigmoid(z):
    """
    Numerically stable sigmoid function.
    Computes σ(z) = 1 / (1 + exp(-z))
    """
    return 1 / (1 + np.exp(-z))


@njit
def predict(theta, xs, batch):
    """
    Predict probabilities for a batch of samples using logistic regression.

    Parameters
    ----------
    theta : 1D array
        Model parameters (theta[0] = bias, theta[1:] = weights).
    xs : 2D array
        Feature matrix of shape (n_samples, n_features).
    batch : 1D array (indices)
        Indices of the current mini-batch.

    Returns
    -------
    1D array
        Predicted probabilities for each sample in the batch.
    """
    bias = theta[0]
    weight = theta[1:]

    # Linear combination: z = b + x·w
    linear = bias + np.dot(xs[batch], weight)

    return sigmoid(linear)


@njit
def predict_single_value(theta, input):
    """
    Predict probability for a *single* feature vector.

    Parameters
    ----------
    theta : 1D array
        Model parameters.
    input : 1D array
        Feature vector.

    Returns
    -------
    float
        Predicted probability.
    """
    bias = theta[0]
    weight = theta[1:]
    linear = bias + np.dot(input, weight)
    return sigmoid(linear)


@njit
def J_logistic(theta, xs, y, batch):
    """
    Compute the logistic regression loss (cross-entropy) for a batch.

    Parameters
    ----------
    theta : 1D array
        Model parameters.
    xs : 2D array
        Feature matrix.
    y : 1D array
        Ground-truth labels (0 or 1).
    batch : 1D array
        Indices for the current mini-batch.

    Returns
    -------
    float
        Sum of logistic loss over the batch.
    """
    h = predict(theta, xs, batch)
    y_batch = y[batch]

    # Small epsilon for numerical stability in log()
    eps = 1e-12

    # Cross-entropy loss (summed)
    loss = -(
        y_batch * np.log(h + eps)
        + (1 - y_batch) * np.log(1 - h + eps)
    )

    return loss.sum()


@njit
def gradient_J_logistic(theta, xs, y, batch):
    """
    Compute the gradient of the logistic loss for a batch.

    Parameters
    ----------
    theta : 1D array
        Model parameters.
    xs : 2D array
        Feature matrix.
    y : 1D array
        Ground-truth labels.
    batch : 1D array
        Indices for the current mini-batch.

    Returns
    -------
    1D array
        Gradient vector with same shape as theta.
        grad[0] = dLoss/dBias
        grad[1:] = dLoss/dWeights
    """
    # Predicted probabilities
    h = predict(theta, xs, batch)
    y_batch = y[batch]

    # Logistic error term
    error = h - y_batch

    # Gradient w.r.t. bias (sum of errors)
    dp = error.sum()

    # Gradient w.r.t. weights (matrix-vector multiplication)
    dw = np.dot(xs[batch].T, error)

    # Combine into full gradient vector
    grad = np.empty(theta.shape)
    grad[0] = dp
    grad[1:] = dw

    return grad