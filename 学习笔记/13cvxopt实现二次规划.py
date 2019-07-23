# cvxopt实现二次规划
# 参考https://blog.csdn.net/jclian91/article/details/79321407
# min f(x)=0.5*x'*P*x + q'*x       '表示转置
# s.t.
#       G*x<=h
#       A*x=b

import numpy as np    # 先导入numpy，安装的版本也要注意，不能直接用pip
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False     # 静默求解过程

Q = 2*matrix([[2, .5], [.5, 1]])
p = matrix([1.0, 1.0])
G = matrix([[-1.0,0.0],[0.0,-1.0]])
h = matrix([0.0,0.0])
A = matrix([1.0, 1.0], (1,2))
b = matrix(1.0)

sol=solvers.qp(Q, p, G, h, A, b)
print(sol['x'])     # 是一个matrix
print(sol['primal objective'])    # 最优值
