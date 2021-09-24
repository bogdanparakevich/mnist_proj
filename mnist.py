import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


data = pd.read_csv('/Users/bogdan_parakevich/mnist_proj/train.csv')

data = np.array(data)
np.random.shuffle(data) # shuffle before splitting into dev and training sets
data = data.T

Y_train = data[0, 1000:]
Y_true = Y_train

X_train = data[1:,1000:]
X_train = X_train / 255.

Y_dev = data[0,:1000]
X_dev = data[1:,:1000]
X_dev = X_dev / 255.

W1 = np.random.random((10, 784)) - 0.5
b1 = np.random.random((10, 1)) - 0.5
W2 = np.random.random((10, 10)) - 0.5
b2 = np.random.random((10, 1)) - 0.5

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def argmax(A):
    return np.argmax(A, 0)

def ReLU_deriv(Z):
    return Z > 0


Z1 = (W1 @ X_train) + b1
A1 = ReLU(Z1)
Z2 = (W2 @ A1) + b2
A2 = softmax(Z2)

Y_true_matrix = np.zeros((41000, 10))
Y_true_matrix[np.arange(41000), Y_true] = 1
Y_true_matrix = Y_true_matrix.T

epochs = 501
alpha = 0.1
m = A2.shape[1]

def accuracy(Y_pred, Y_true_matrix):
    return np.sum(Y_pred == Y_true_matrix) / Y_pred.size

for i in range(epochs):
    Z1 = (W1 @ X_train) + b1
    A1 = ReLU(Z1)
    Z2 = (W2 @ A1) + b2
    A2 = softmax(Z2)

    dloss_dZ2 = A2 - Y_true_matrix

    dloss_dW2 = (1 / m) * dloss_dZ2 @ A1.T
    dloss_db2 = (1 / m) * np.sum(dloss_dZ2)

    dloss_dZ1 = W2.T @ dloss_dZ2 * ReLU_deriv(Z1)

    dloss_dW1 = (1 / m) * dloss_dZ1 @ X_train.T
    dloss_db1 = (1 / m) * np.sum(dloss_dZ1)

    W1 = W1 - alpha * dloss_dW1
    b1 = b1 - alpha * dloss_db1    
    W2 = W2 - alpha * dloss_dW2  
    b2 = b2 - alpha * dloss_db2
    if i % 100 == 0:
            print("Iteration: ", i)
            Y_pred = argmax(A2)
            print(accuracy(Y_pred, Y_true))


def test_madel(W1,b1,W2,b2,X):
    Z1 = (W1 @ X) + b1
    A1 = ReLU(Z1)
    Z2 = (W2 @ A1) + b2
    A2 = softmax(Z2)
    A2_dev = A2
    return A2_dev

A2_dev = test_madel(W1,b1,W2,b2,X_dev)
A2_dev = argmax(A2_dev)


accuracy_dev = accuracy(A2_dev, Y_dev)
print('accuracy_dev =', accuracy_dev)










