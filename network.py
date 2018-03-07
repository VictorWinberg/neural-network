import numpy as np
import matplotlib.pyplot as plt

# sigmoid function
def nonlin(x, deriv=False):
    if(deriv == True):
        return x * (1-x)
    return 1 / (1 + np.exp(-x))

# input dataset
X = np.array([
    [1,1,1,0,0],
    [1,1,0,1,1],
    [1,0,1,0,1],
    [0,1,1,0,0],
    [0,1,0,0,1],
    [0,0,1,1,0]
])
    
# output dataset            
y = np.array([[1,1,1,0,0,0]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# bias value that shifts the activation function
bias = 1

# initialize weights randomly with mean 0
syn0 = 2 * np.random.random((X[0].size, X.size)) - bias
syn1 = 2 * np.random.random((X.size, y[0].size)) - bias

# training
for iter in range(10000):

    # forward propagation with layers
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    # backpropagation for layer 2
    l2_error = y - l2

    # multiply how much we missed by the slope
    # of the sigmoid at the values in layer 2
    l2_delta = l2_error * nonlin(l2, deriv=True)

    # backpropagation for layer 1
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1, deriv=True)

    # update weights
    syn1 += np.dot(l1.T, l2_delta)
    syn0 += np.dot(l0.T, l1_delta)

print("Error:", np.mean(np.abs(l2_error)))

X_test = np.array([
    [0,1,0,0,0],
    [0,0,0,1,1],
    [0,1,1,1,0],
    [0,0,1,0,1],
    [0,0,0,0,1],
    [1,1,1,0,0],
    [1,0,1,0,0],
    [1,0,0,1,0],
    [1,0,0,0,1],
    [1,0,0,0,0]
])

l0_test = X_test
l1_test = nonlin(np.dot(l0_test, syn0))
l2_test = nonlin(np.dot(l1_test, syn1))

correct = np.sum(np.round(l2_test, 1) == np.array([[0,0,0,0,0,1,1,1,1,1]]).T)
print("Correct:", correct, "/", X_test.shape[0])
