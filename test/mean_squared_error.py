import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from tqdm import tqdm

from middleware.core.variable import Variable
import middleware.functions.components.arithmetic as F

np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))

def predict(x):
  y = F.matmul(x, W) + b
  return y

def mean_squared_error(x0, x1):
  diff = x0 - x1
  return F.sum(diff ** 2) / len(diff)


lr = 0.1
iters = 15

for i in range(iters):
  y_pred = predict(x)
  loss = mean_squared_error(y, y_pred)

  W.cleargrad()
  b.cleargrad()
  loss.backward()

  W.data -= lr * W.grad.data
  b.data -= lr * b.grad.data
  print(W, b, loss)

  plt.scatter(x.data, y.data, s=10)
  plt.xlabel('x')
  plt.ylabel('y')
  y_pred = predict(x)
  plt.plot(x.data, y_pred.data, color='r')
  plt.show(block=False)
  plt.pause(0.5)
  plt.close()
