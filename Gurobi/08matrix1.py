# matrix1模型，采用稀疏矩阵存储模型
# This example formulates and solves the following simple MIP model
# using the matrix API:
#  maximize
#        x +   y + 2 z
#  subject to
#        x + 2 y + 3 z <= 4
#        x +   y       >= 1
#        x, y, z binary

import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB

try:

    # (1)Create a new model
    m = gp.Model("matrix1")

    # (2)Create variables
    x = m.addMVar(shape=3, vtype=GRB.BINARY, name="x")  #3个决策变量

    # (3)Set objective
    obj = np.array([1.0, 1.0, 2.0])  # 目标函数系数矩阵
    m.setObjective(obj @ x, GRB.MAXIMIZE)

    # (4)Build (sparse) constraint matrix稀疏矩阵，松弛矩阵
    data = np.array([1.0, 2.0, 3.0, -1.0, -1.0])  # 约束条件中非零的5个数
    row = np.array([0, 0, 0, 1, 1])  # 5个元素在A中的行索引
    col = np.array([0, 1, 2, 0, 1])  # 5个元素在A中的列索引

    A = sp.csr_matrix((data, (row, col)), shape=(2, 3))

    # (5)Build rhs vector等号右侧
    rhs = np.array([4.0, -1.0])

    # (6)Add constraints
    m.addConstr(A @ x <= rhs, name="c")

    # Optimize model
    m.optimize()

    print(x.X)
    print('Obj: %g' % m.objVal)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')
