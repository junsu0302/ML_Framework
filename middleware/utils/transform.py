import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np

from middleware.core.variable import Variable

# TODO : Variable 타입으로 변환
def as_variable(obj):
  if isinstance(obj, Variable):
    return obj
  return Variable(obj)

# TODO : 배열 형태로 변환
def as_array(x):
  if np.isscalar(x):
    return np.array(x)
  return x