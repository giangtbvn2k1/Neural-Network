from network.network import NetWork
from layer.full_conneted_layer import FCLayer
from layer.Activation_layer import ActivationLayer
import numpy as np


def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    z[z<0] = 0
    z[z>0] = 1
    return z

def loss(y, y_predict):
    return 0.5*(y_predict - y)**2

def loss_prime(y, y_predict):
    return y_predict - y

X_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[0]], [[0]], [[1]]])

net = NetWork()
net.add(FCLayer((1, 2), (1, 3)))
net.add(ActivationLayer((1, 3), (1, 3), relu, relu_prime))
net.add(FCLayer((1, 3), (1, 1)))
net.add(ActivationLayer((1, 1), (1, 1), relu, relu_prime))

net.setup(loss, loss_prime)

net.fit(X_train, y_train, learning_rate=0.01, epochs= 5000)

print(net.predict([[1,1]]))