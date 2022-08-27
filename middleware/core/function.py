import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import weakref

from middleware.core.variable import Variable
from middleware.utils.transform import as_variable, as_array
import middleware.config.config as Mode

class Function:
  def __call__(self, *inputs): # 가변 인수 호출
    inputs = [as_variable(x) for x in inputs]

    # TODO : 순전파 계산
    xs = [x.data for x in inputs] # 입력받은 데이터
    ys = self.forward(*xs) # 구체적 계산은 forward 메서드에서 수행
                           # 리스트를 전달할 때 낱개로 풀어서 전달
    if not isinstance(ys, tuple):
      ys = (ys,) # 튜플이 아닌 경우 튜플로 변경
    outputs = [Variable(as_array(y)) for y in ys] # 출력할 데이터

    if Mode.Config.enable_backprop: # 역전파 모드
      self.generation = max([x.generation for x in inputs]) # 세대 설정

      # TODO : 연결 생성
      for output in outputs:
        output.set_creator(self) # 출력 변수에 생성자 설정
      self.inputs = inputs # 입력 변수 보관
      self.outputs = [weakref.ref(output) for output in outputs] # 생성자에 대한 출력 저장
    
    return outputs if len(outputs) > 1 else outputs[0] # 가변 인수 출력

  def forward(self, xs):
    raise NotImplementedError()

  def backward(self, gys):
    raise NotImplementedError()
