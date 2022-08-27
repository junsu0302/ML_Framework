import middleware.core
import middleware.config
import middleware.functions
import middleware.data
import middleware.models
import middleware.utils

from middleware.functions.components.arithmetic import setup_arithmetic

setup_arithmetic()

"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from middleware.core.variable import Variable
from middleware.core.function import Function
from middleware.config.config import Config
from middleware.config.config import using_config
from middleware.utils.transform import as_array, as_variable

# ! Functions
# ? Components
from middleware.functions.components.arithmetic import Add, add
from middleware.functions.components.arithmetic import Mul, mul
from middleware.functions.components.arithmetic import Sub, sub
from middleware.functions.components.arithmetic import Div, div
from middleware.functions.components.arithmetic import Neg, neg
from middleware.functions.components.arithmetic import Pow, pow
from middleware.functions.components.arithmetic import Square, square
from middleware.functions.components.arithmetic import Exp, exp
from middleware.functions.components.arithmetic import Sum, sum
from middleware.functions.components.arithmetic import MatMul, matmul
from middleware.functions.components.arithmetic import setup_arithmetic

from middleware.functions.components.transform import Reshape, reshape
from middleware.functions.components.transform import Transpose, transpose
from middleware.functions.components.transform import BroadcaseTo, broadcast_to
from middleware.functions.components.transform import SumTo, sum_to
from middleware.functions.components.transform import Linear, linearm, linear_simple

from middleware.functions.components.trigonmentric import Sin, sin
from middleware.functions.components.trigonmentric import Cos, cos
from middleware.functions.components.trigonmentric import Tanh, tanh

from middleware.functions.loss import mean_squared_error

setup_arithmetic()
"""