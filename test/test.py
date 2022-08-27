import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from tqdm import tqdm

from middleware.core.variable import Variable
import middleware.functions.activation as F
import middleware.functions.loss as FL
import middleware.models.layers as L
from middleware.models.layers import Layer

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

l1 = L.Linear(10)
l2 = L.Linear(1)


def predict(x):
    y = l1(x)
    y = F.sigmoid_simple(y)
    y = l2(y)
    return y


lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = FL.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)
        plt.scatter(x, y, s=10)
        plt.xlabel('x')
        plt.ylabel('y')
        t = np.arange(0, 1, .01)[:, np.newaxis]
        y_pred = predict(t)
        plt.plot(t, y_pred.data, color='r')
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()