import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import shuffle


data = pd.read_csv('/Users/bogdan_parakevich/mnist_proj/train.csv')
data = np.array(data)

epochs = 26
alpha = 0.1
momentum = 0.9
change_W1 = 0.001
change_b1 = 0.001
change_W2 = 0.001
change_b2 = 0.001

np.random.shuffle(data)
data = data.T
n_test = 4200
train_data = data[:, n_test:]
m = train_data.shape[1]
Y_train = train_data[0]
X_train = train_data[1:]
X_train = X_train / 255.
Y_true_matrix = np.zeros((m, 10))
Y_true_matrix[np.arange(m), Y_train] = 1
Y_true_matrix = Y_true_matrix.T
n_batchs = 32
X_batches = np.array_split(X_train, n_batchs, axis=1)
Y_batches = np.array_split(Y_true_matrix, n_batchs, axis=1)

test_data = data[:,:n_test]
Y_test = data[0,:n_test]
X_test = data[1:,:n_test]
X_test = X_test / 255.
Y_test_matrix = np.zeros((n_test, 10))
Y_test_matrix[np.arange(n_test), Y_test] = 1
Y_test_matrix = Y_test_matrix.T

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

def accuracy(pred, true):
    return np.sum(pred == true) / true.size

def test_model(W1, b1, W2, b2, X):
    Z1 = (W1 @ X) + b1
    A1 = ReLU(Z1)
    Z2 = (W2 @ A1) + b2
    A2 = softmax(Z2)
    return A2

x_plot_train = []
y_plot_train = []
x_plot_test = []
y_plot_test = []

for i in range(epochs):
    
    X_batches, Y_batches = shuffle(X_batches, Y_batches, random_state=n_batchs)
    
    for k in range(n_batchs):
        m_batch = Y_batches[k].shape[1]
        Z1 = (W1 @ X_batches[k]) + b1
        A1 = ReLU(Z1)
        Z2 = (W2 @ A1) + b2
        A2 = softmax(Z2)

        dZ2 = A2 - Y_batches[k]
        dW2 = 1 / m_batch * dZ2 @ A1.T
        db2 = 1 / m_batch * np.sum(dZ2)
        dZ1 = W2.T @ dZ2 * ReLU_deriv(Z1)
        dW1 = 1 / m_batch * dZ1 @ X_batches[k].T
        db1 = 1 / m_batch * np.sum(dZ1)

        change_W1 = (momentum * change_W1) - (alpha * dW1)
        change_b1 = (momentum * change_b1) - (alpha * db1)
        change_W2 = (momentum * change_W2) - (alpha * dW2)
        change_b2 = (momentum * change_b2) - (alpha * db2)

        W1 = W1 + change_W1
        b1 = b1 + change_b1
        W2 = W2 + change_W2
        b2 = b2 + change_b2
        
        e = 1e-6
        loss = - (1 / m_batch) * np.sum(Y_batches[k] * np.log(A2))
        
        A2_test = test_model(W1, b1, W2, b2, X_test)
        loss_test = - (1 / n_test) * np.sum(Y_test_matrix * np.log(A2_test))
        
        Y_pred = argmax(A2)
        Y_true = argmax(Y_batches[k])
        
        Y_pred_test = argmax(A2_test)
        Y_true_test = argmax(Y_test_matrix)
        
        k = k + 1
        
    if i % 5 == 0:
        print('epochs:', i)
        print('loss_train:', loss)
        print('loss_test:', loss_test)
        print('accuracy_train:', (np.sum(Y_pred == Y_true) / Y_true.size))
        print('accuracy_test:', (np.sum(Y_pred_test == Y_true_test) / Y_true_test.size))
        
    x_plot_train.append(i)
    y_plot_train.append(loss)
    
    x_plot_test.append(i)
    y_plot_test.append(loss_test)


plt.plot(x_plot_train, y_plot_train, "r", label="loss_train")
plt.plot(x_plot_test, y_plot_test, "g", label="loss_test")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.show() 









