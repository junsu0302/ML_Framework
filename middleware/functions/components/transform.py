import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np

import middleware.utils.functions as Utils
from middleware.core.function import Function
from middleware.utils.transform import as_variable

# TODO : 행렬 형상 변환
class Reshape(Function):
  def __init__(self, shape):
    self.shape = shape

  def forward(self, x):
    self.x_shape = x.shape
    return x.reshape(self.shape)
  
  def backward(self, gy):
    return reshape(gy, self.x_shape)

def reshape(x, shape):
  if x.shape == shape:
    return as_variable(x)
  return Reshape(shape)(x)

# TODO : 행렬 전치
class Transpose(Function):
  def forward(self, x):
    return np.transpose(x)

  def backward(self, gy):
    return transpose(gy)

def transpose(x):
  return Transpose()(x)

# TODO : 입력 변수와 형상이 같아지도록 기울기 원소 복제
class BroadcaseTo(Function):
  def __init__(self, shape):
    self.shape = shape # 목표 형상 저장

  def forward(self, x):
    self.x_shape = x.shape # 입력 변수 형상 저장
    return np.broadcast_to(x, self.shape) # 목표 형상처럼 복제  

  def backward(self, gy):
    return sum_to(gy, self.x_shape)

def broadcast_to(x, shape): 
  if x.shape == shape:
    return as_variable(x)
  return BroadcaseTo(shape)(x)

# TODO : 입력 변수와 형상이 같은 상태로 원소의 합 계산
class SumTo(Function):  
  def __init__(self, shape):
    self.shape = shape # 목표 형삳 저장

  def forward(self, x):
    self.x_shape = x.shape # 입력 변수 형상 저장
    return Utils.sum_to_shape(x, self.shape) 

  def backward(self, gy):
    return broadcast_to(gy, self.x_shape) # 목표 형상처럼 복제

def sum_to(x, shape):
  if x.shape == shape:
    return as_variable(x)
  return SumTo(shape)(x)
