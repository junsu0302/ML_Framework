import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np

import middleware
from middleware.core.variable import Variable
from middleware.core.function import Function
from middleware.utils.transform import as_array
import middleware.utils.functions as Utils
import middleware.functions.components.transform as T


class Add(Function):
  def forward(self, x0, x1):
    self.x0_shape, self.x1_shape = x0.shape, x1.shape
    return x0 + x1

  def backward(self, gy):
    gx0, gx1 = gy, gy
    if self.x0_shape != self.x1_shape:
      gx0 = middleware.functions.components.transform.sum_to(gx0, self.x0_shape)
      gx1 = middleware.functions.components.transform.sum_to(gx1, self.x1_shape)
    return gx0, gx1

def add(x0, x1):
  x1 = as_array(x1)
  return Add()(x0, x1)

class Mul(Function):
  def forward(self, x0, x1):
    self.x0_shape, self.x1_shape = x0.shape, x1.shape
    return x0 * x1

  def backward(self, gy):
    x0, x1 = self.inputs
    gx0, gx1 = gy * x1, gy * x0
    if self.x0_shape != self.x1_shape:
      gx0 = middleware.functions.components.arithmetic.sum_to(gx0, self.x0_shape)
      gx1 = middleware.functions.components.arithmetic.sum_to(gx1, self.x1_shape)
    return gx0, gx1

def mul(x0, x1):
  x1 = as_array(x1)
  return Mul()(x0, x1)

class Sub(Function):
  def forward(self, x0, x1):
    self.x0_shape, self.x1_shape = x0.shape, x1.shape
    return x0 - x1

  def backward(self, gy):
    gx0, gx1 = gy, -gy
    if self.x0_shape != self.x1_shape:
      gx0 = middleware.functions.components.arithmetic.sum_to(gx0, self.x0_shape)
      gx1 = middleware.functions.components.arithmetic.sum_to(gx1, self.x1_shape)
    return gx0, gx1

def sub(x0, x1):
  x1 = as_array(x1)
  return Sub()(x0, x1)

def rsub(x0, x1):
  x1 = as_array(x1)
  return sub(x1, x0)


class Div(Function):
  def forward(self, x0, x1):
    self.x0_shape, self.x1_shape = x0.shape, x1.shape
    return x0 / x1

  def backward(self, gy):
    x0, x1 = self.inputs
    gx0, gx1 = gy / x1, gy * (-x0 / x1 ** 2)
    if self.x0_shape != self.x1_shape:
      gx0 = middleware.functions.components.transform.sum_to(gx0, self.x0_shape)
      gx1 = middleware.functions.components.transform.sum_to(gx1, self.x1_shape)
    return gx0, gx1

def div(x0, x1):
  x1 = as_array(x1)
  return Div()(x0, x1)

def rdiv(x0, x1):
  x1 = as_array(x1)
  return div(x1, x0)

class Neg(Function):
  def forward(self, x):
    return -x

  def backward(self, gy):
    return -gy

def neg(x):
  return Neg()(x)

class Pow(Function):
  def __init__(self, c):
    self.c = c

  def forward(self, x):
    return x ** self.c

  def backward(self, gy):
    x, = self.inputs
    c = self.c
    return c * x ** (c - 1) * gy

def pow(x, c):
  return Pow(c)(x)

class Square(Function):
  def forward(self, x):
    return x ** 2

  def backward(self, gy):
    x = self.outputs[0]() 
    return 2 * x * gy

def square(x):
  return Square()(x)

class Exp(Function):
  def forward(self, x):
    return np.exp(x)

  def backward(self, gy):
    x = self.outputs[0]()
    return x * gy

def exp(x):
    return Exp()(x)


# TODO : 행렬의 합
class Sum(Function):
  def __init__(self, axis, keepdims):
    self.axis = axis # 축(차원)
    self.keepdims = keepdims # 축(차원) 수를 똑같이 유지할 것인가

  def forward(self, x):
    self.x_shape = x.shape
    return x.sum(axis=self.axis, keepdims=self.keepdims)

  def backward(self, gy):
    gy = Utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
    return T.broadcast_to(gy, self.x_shape)

def sum(x, axis=None, keepdims=False):
  return Sum(axis, keepdims)(x)

# TODO : 행렬의 곱
class MatMul(Function):
  def forward(self, x, W):
    return x.dot(W)

  def backward(self, gy):
    x, W = self.inputs
    gx = matmul(gy, W.transpose())
    gW = matmul(x.transpose(), gy)
    return gx, gW

def matmul(x, W):
  return MatMul()(x, W)

def setup_arithmetic():
  Variable.__add__ = add
  Variable.__radd__ = add
  Variable.__mul__ = mul
  Variable.__rmul__ = mul
  Variable.__neg__ = neg
  Variable.__sub__ = sub
  Variable.__rsub__ = rsub
  Variable.__truediv__ = div
  Variable.__rtruediv__ = rdiv
  Variable.__pow__ = pow