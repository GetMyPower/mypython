# cvxoptʵ�ֶ��ι滮
# �ο�https://blog.csdn.net/jclian91/article/details/79321407
# min f(x)=0.5*x'*P*x + q'*x       '��ʾת��
# s.t.
#       G*x<=h
#       A*x=b

import numpy as np    # �ȵ���numpy����װ�İ汾ҲҪע�⣬����ֱ����pip
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False     # ��Ĭ������

Q = 2*matrix([[2, .5], [.5, 1]])
p = matrix([1.0, 1.0])
G = matrix([[-1.0,0.0],[0.0,-1.0]])
h = matrix([0.0,0.0])
A = matrix([1.0, 1.0], (1,2))
b = matrix(1.0)

sol=solvers.qp(Q, p, G, h, A, b)
print(sol['x'])     # ��һ��matrix
print(sol['primal objective'])    # ����ֵ
