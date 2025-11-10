import numpy as np
import plotly.express as px
from numba import njit

x_train = np.arange(0.0, 1.0, 0.025)
y_train = 0.4 + x_train * 0.55 + np.random.randn(x_train.shape[0])*0.2


@njit
def predict(theta, xs, batch):
    bias = theta[0]
    weight = theta[1:] 
    return bias + np.dot(xs[batch], weight)

@njit
def J_squared_residual(theta, xs, y, batch):
    h = predict(theta, xs, batch)
    sr = ((h - y[batch])**2).sum()    
    return sr

@njit
def gradient_J_squared_residual(theta, xs, y, batch):
    h = predict(theta, xs, batch) 
    #sum(x_i^2) = x^T * x (with x = h - y and h = X*0)
    # ----> J(0) = (h-y)^T * (h-y)
    # grad = dJ/d0 ------> dJ/d0 = xs^t * (h-y)
    # this is gradient computed with some fancy math I probably cant remember 
    # simplfied: calculates partial derivatives from L ------> 0 (backward propagation)
    # dJ/d0 = dJ/dh * dh/d0
    grad = np.dot(xs[batch].transpose(), (h - y[batch])) 
    return grad


# the dataset (already augmented so that we get a intercept coef)
# remember: augmented x -> we add a colum of 1's instead of using a bias term.
#data_x = np.array([[0.5], [1.0], [2.0]])  # (3,1)
#data_y = np.array([[1.0], [1.5], [2.5]]) # (3,1)
data_x = x_train
data_y = y_train
# make to collumn vektor
data_x = np.array(data_x).reshape(-1, 1)
data_y = np.array(data_y).reshape(-1, 1)
n_features = data_x.shape[1]

# variables we need 
theta = np.zeros((n_features + 1, 1)) #(2,1)
learning_rate = 0.1
m = data_x.shape[0]
batch_size = 39


# run GD
j_history = []
n_iters = 30
for it in range(n_iters):
    batch = np.random.randint(0, batch_size + 1, size=batch_size)
    j = J_squared_residual(theta, data_x, data_y, batch)
    j_history.append(j)
    
    theta = theta - (learning_rate * (1/m) * gradient_J_squared_residual(theta, data_x, data_y, batch))
    
print("theta shape:", theta.shape)

# append the final result.
j = J_squared_residual(theta, data_x, data_y, batch)
j_history.append(j)
print("The L2 error is: {:.2f}".format(j))


# find the L1 error.
y_pred = predict(theta, data_x, batch)
l1_error = np.abs(y_pred - data_y[batch]).sum()
print("The L1 error is: {:.2f}".format(l1_error))


# Find the R^2 
# if the data is normalized: use the normalized data not the original data (task 3 hint).
# https://en.wikipedia.org/wiki/Coefficient_of_determination
u = ((data_y[batch] - y_pred)**2).sum()
v = ((data_y[batch] - data_y[batch].mean())** 2).sum()
print("R^2: {:.2f}".format(1 - (u/v)))


# plot the result
fig = px.line(j_history, title="J(theta) - Loss History")
fig.show()
