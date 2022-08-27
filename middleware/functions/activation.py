import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from middleware.core.function import Function
from middleware.utils.transform import as_variable
from middleware.functions.components.arithmetic import exp
from middleware.functions.components.trigonmentric import tanh

def sigmoid_simple(x):
  x = as_variable(x)
  return  1 / (1 + exp(-x))

class Sigmoid(Function):
  def forward(self, x):
    return tanh(x * 0.5) * 0.5 + 0.5

  def backward(self, gy):
    y = self.outputs[0]()
    return gy * y * (1 - y)


def sigmoid(x):
  return Sigmoid()(x)