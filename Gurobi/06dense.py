# dense全矩阵式输入模型
# This example formulates and solves the following simple QP model:
#
#    minimize    x + y + x^2 + x*y + y^2 + y*z + z^2
#    subject to  x + 2 y + 3 z >= 4
#                x +   y       >= 1
#                x, y, z non-negative
#
# The example illustrates the use of dense matrices to store A and Q
# (and dense vectors for the other relevant data).  We don't recommend
# that you use dense matrices, but this example may be helpful if you
# already have your data in this format.

import sys
import gurobipy as gp
from gurobipy import GRB


def dense_optimize(rows, cols, c, Q, A, sense, rhs, lb, ub, vtype,
                   solution):

    model = gp.Model()

    # (1)Add variables to model, 共cols个决策变量
    vars = []
    for j in range(cols):
        vars.append(model.addVar(lb=lb[j], ub=ub[j], vtype=vtype[j]))

    # (2)Populate A matrix, A.shape=(rows, cols)
    for i in range(rows):
        expr = gp.LinExpr()
        for j in range(cols):
            if A[i][j] != 0:
                expr += A[i][j]*vars[j]
        model.addConstr(expr, sense[i], rhs[i])

    # (3)Populate objective，二次项系数矩阵Q.shape=(cols, cols),一次项系数矩阵c.shape=(3,)
    obj = gp.QuadExpr()
    for i in range(cols):
        for j in range(cols):
            if Q[i][j] != 0:
                obj += Q[i][j]*vars[i]*vars[j]
    for j in range(cols):
        if c[j] != 0:
            obj += c[j]*vars[j]
    model.setObjective(obj)

    # (4)Solve
    model.optimize()

    # (5)Write model to a file
    model.write('dense.lp')

    if model.status == GRB.OPTIMAL:
        x = model.getAttr('x', vars)
        for i in range(cols):
            solution[i] = x[i]    # 返回计算结果
        return True
    else:
        return False


# Put model data into dense matrices

c = [1, 1, 0]   # 目标函数中的一次项，x和y
Q = [[1, 1, 0], [0, 1, 1], [0, 0, 1]]  # 目标函数中的二次项系数矩阵
A = [[1, 2, 3], [1, 1, 0]]   # 约束条件的系数矩阵
sense = [GRB.GREATER_EQUAL, GRB.GREATER_EQUAL]  # 约束条件的大小关系
rhs = [4, 1]   # 约束条件的等号右侧
lb = [0, 0, 0]  # 三个决策变量的下限为0
ub = [GRB.INFINITY, GRB.INFINITY, GRB.INFINITY]  # 三个决策变量的上限
vtype = [GRB.CONTINUOUS, GRB.CONTINUOUS, GRB.CONTINUOUS]  # 连续型变量
sol = [0]*3   # 可能是初解，但是最终解将保存在sol中

# Optimize

success = dense_optimize(2, 3, c, Q, A, sense, rhs, lb, ub, vtype, sol)

if success:
    print('x: %g, y: %g, z: %g' % (sol[0], sol[1], sol[2]))
