import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from middleware.core.function import Function
from middleware.functions.components.arithmetic import matmul
from middleware.functions.components.transform import sum_to

def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None
    return y

class Linear(Function):
  def forward(self, x, W, b):
    y = x.dot(W)
    if b is not None:
      y += b
    return y

  def backward(self, gy):
    x, W, b = self.inputs
    gb = None if b.data is None else sum_to(gy, b.shape)
    gx = matmul(gy, W.transpose())
    gW = matmul(x.transpose(), gy)
    return gx, gW, gb


def linear(x, W, b=None):
  return Linear()(x, W, b)