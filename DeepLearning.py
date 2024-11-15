import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

print("Dimension de X:", x.shape)
print("Dimension de Y:", y.shape)

def init(X):
    w = np.random.randn(x.shape[1], 1)
    b = np.random.randn(1)
    return (w, b)

w, b = init(x)

def modele(x, w, b):
    z = x.dot(w) + b
    a = 1 / (1 + np.exp(-z))
    return a

a = modele(x, w, b)

def log_loss(a, y):
    return 1 / len(y) * np.sum(-y * np.log(a) - (1- y) * np.log(1 - a))

def gradients(a, x, y):
    dW = 1 / len(y) * np.dot(x.T, a - y)
    dB = 1 / len(y) * np.sum(a - y)
    return (dW, dB)

dW, dB = gradients(a, x, y) 

def update(dW, dB, w, b, learning_rate):
    w = w - learning_rate * dW
    b = b - learning_rate * dB
    return (w, b)

def artificial_neuron(x, y, learning_rate=0.1, iteration=100):
    # init W, B
    w,b = init(x)

    loss = []

    for i in range(iteration):
        a = modele(x, w, b)
        loss.append(log_loss(a, y))
        dW, dB = gradients(a, x, y)
        w, b = update(dW, dB, w, b, learning_rate)
    plt.plot(loss)
    plt.show()

artificial_neuron(x, y)
