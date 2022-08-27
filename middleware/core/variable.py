import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import middleware
from middleware.config.config import Config, using_config, no_grad

class Variable:
  __array_priority__ = 200
  def __init__(self, data, name=None):
    # TODO : ndarray 타입 외 인스턴스 입력시 에러
    if data is not None:
      if not isinstance(data, np.ndarray): 
        raise TypeError('{} is not supported'.format(type(data)))
		
    # TODO : 변수
    self.data = data # 데이터 저장
    self.name = name # 이름 저장
    self.grad = None # 기울기 저장
    self.creator = None # 생성자 저장
    self.generation = 0 # 세대 저장

  # TODO : 추가 기능
  @property
  def shape(self):
    return self.data.shape

  @property
  def ndim(self): # 차원 수
    return self.data.ndim

  @property
  def size(self): # 원소 수
    return self.data.size

  @property
  def dtype(self): # 데이터 타입
    return self.data.dtype

  def __len__(self): # 길이
    return len(self.data)

  def __repr__(self): # 내용
    if self.data is None:
      return 'variable(None)'
    p = str(self.data).replace('\n', '\n' + ' ' * 9)
    return 'variable(' + p + ')'

  # TODO : 전방 함수 정의
  def set_creator(self, func): 
    self.creator = func # creator 설정
    self.generation = func.generation + 1 # 세대 설정
  
  # TODO : 미분값 초기화
  def cleargrad(self):
    self.grad = None
  
  # TODO : 행렬 형상 변환
  def reshape(self, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
      shape = shape[0]
    return middleware.functions.components.transform.reshape(self, shape)
  
  # TODO : 행렬 전치
  def transpose(self):
    return middleware.functions.components.transform.transpose(self)

  # TODO : 행렬 총합
  def sum(self, axis=None, keepdims=False): 
    return middleware.functions.components.arithmetic.sum(self, axis, keepdims)

  # TODO : 역전파
  def backward(self, retain_grad=False, create_graph=False):
    if self.grad is None: # 최초 기울기 값에 자동으로 ndarray 타입 인스턴스 생성
      self.grad = Variable(np.ones_like(self.data))

    funcs = []
    seen_set = set() # 함수 중복 추가 방지

    # TODO : 함수 리스트를 세대 순으로 정렬
    def add_func(f): 
      if f not in seen_set:
        funcs.append(f)
        seen_set.add(f)
        funcs.sort(key=lambda x: x.generation)

    add_func(self.creator)

    while funcs:
      f = funcs.pop() # 함수 가져옴
      gys = [output().grad for output in f.outputs] # 함수의 출력값 저장
	
      # TODO : 역전파 계산
      with using_config('enable_backprop', create_graph):
        gxs = f.backward(*gys) # 함수의 역전파 호출
        if not isinstance(gxs, tuple):
          gxs = (gxs,) # 튜플이 아닌 경우 튜플로 변경

        # TODO : 역전파 저장
        for x, gx in zip(f.inputs, gxs): # 역전파로 전달되는 미분값 저장
          if x.grad is None:
            x.grad = gx 
          else:
            x.grad = x.grad + gx 

          if x.creator is not None:
            add_func(x.creator) # 바로 앞 함수를 리스트에 추가

      # TODO : 각 함수의 출력 변수의 미분값을 유지하지 않도록 설정
      if not retain_grad:
        for y in f.outputs:
          y().grad = None