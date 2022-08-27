import sys, os
from unittest.mock import NonCallableMagicMock
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import weakref

from middleware.core.parameter import Parameter
import middleware.functions.lenear as F


# TODO : 계층 정보 저장
class Layer:
  def __init__(self):
    self._params = set() # Layer 인스턴스에 속한 매개변수 보관

  def __setattr__(self, name, value): # ? 변수 설정 시 호출
    if isinstance(value, (Parameter, Layer)):
      self._params.add(name)
    super().__setattr__(name, value) # 인스턴스 변수를 이름과 값의 형태로 재정의

  def __call__(self, *inputs):
    outputs = self.forward(*inputs)
    if not isinstance(outputs, tuple):
      outputs = (outputs, )
    self.inputs = [weakref.ref(x) for x in inputs]
    self.outputs = [weakref.ref(y) for y in outputs]
    return outputs if len(outputs) > 1 else outputs[0]

  def forward(self, inputs):
    raise NotImplementedError()

  def params(self): # Parameter 인스턴스들을 꺼냄
    for name in self._params:
      obj = self.__dict__[name]

      if isinstance(obj, Layer): # Layer에서 매개변수 꺼내기
        yield from obj.params()
      else:
        yield obj 

  def cleargrads(self): # 모든 매개변수 기울기 재설정
    for param in self.params():
      param.cleargrad()

class Linear(Layer):
  def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
    super().__init__()
    self.in_size = in_size # 입력 크기
    self.out_size = out_size # 출력 크기
    self.dtype = dtype # 연산 타입

    self.W = Parameter(None, name="W") # 가중치를 Parameter 인스턴수로 설정

    if self.in_size is not None: # in_size가 지정되어 있지 않다면 나중으로 연기
      self._init_W()

    if nobias: # 편향을 Parameter 인스턴수로 설정
      self.b = None
    else:
      self.b = Parameter(np.zeros(out_size, dtype=dtype), name="b") 

  def _init_W(self): # 가중치 초기화
    I, O = self.in_size, self.out_size
    W_data = np.random.randn(I, O).astype(self.dtype) + np.sqrt(1 / I)
    self.W.data = W_data

  def forward(self, x):
    if self.W.data is None: # 데이터를 흘려보내는 시점에 가중치 초기화
      self.in_size = x.shape[1]
      self._init_W()

    return F.linear(x, self.W, self.b)
