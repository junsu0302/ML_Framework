import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from middleware.core.variable import Variable
import middleware.functions.activation as F
import middleware.functions.loss as L
import middleware.functions.lenear as T

np.random.seed(0)
x = np.random.rand(100, 1)
y = ((np.sin(2 * np.pi * x) + np.random.rand(100, 1)) * np.cos(2 * np.pi * x))*np.tanh(2 * np.pi * x)*np.tanh(2 * np.pi * x)

I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))

list = []

def predict(x):
  y = T.linear(x, W1, b1)
  y = F.sigmoid_simple(y)
  y = T.linear(y, W2, b2)
  return y

lr = 0.3
iters = 300000

for i in tqdm(range(iters), desc="학습 진행률", leave=True):
  y_pred = predict(x)
  loss = L.mean_squared_error(y, y_pred)

  W1.cleargrad()
  b1.cleargrad()
  W2.cleargrad()
  b2.cleargrad()
  loss.backward()

  W1.data -= lr * W1.grad.data
  b1.data -= lr * b1.grad.data
  W2.data -= lr * W2.grad.data
  b2.data -= lr * b2.grad.data

  if i % 30000 == 0:
    list.append(loss)
    plt.scatter(x, y, s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    t = np.arange(0, 1, .01)[:, np.newaxis]
    y_pred = predict(t)
    plt.plot(t, y_pred.data, color='r')
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()

for i in range(10):
  print(i, "번째 기울기", list[i])