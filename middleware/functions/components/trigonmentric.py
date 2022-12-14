import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np

from middleware.core.function import Function

class Sin(Function):
  def forward(self, x):
    return np.sin(x)

  def backward(self, gy):
    x, = self.inputs
    return gy * cos(x)

def sin(x):
  return Sin()(x)

class Cos(Function):
  def forward(self, x):
    return np.cos(x)

  def backward(self, gy):
    x, = self.inputs
    return gy * -sin(x)

def cos(x):
  return Cos()(x)

class Tanh(Function):
  def forward(self, x):
    return np.tanh(x)

  def backward(self, gy):
    y = self.outputs[0]()
    return gy * (1 - y * y)

def tanh(x):
  return Tanh()(x)