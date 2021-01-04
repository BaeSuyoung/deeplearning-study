import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.random.randn(10, 2)
w1 = np.random.randn(2, 4)
b1 = np.random.randn(4)
w2 = np.random.randn(4, 3)
b2 = np.random.randn(3)

h = np.matmul(x, w1) + b1
a = sigmoid(h)
s = np.matmul(a, w2) + b2

print(h)
print(s)
