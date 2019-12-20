# PWL分段线性函数，约束条件中存在分段线性化函数
# maximize
#        sum c[j] * x[j]
#  subject to
#        sum A[i,j] * x[j] <= 0,  for i = 0, ..., m-1
#        sum y[j] <= 3
#        y[j] = pwl(x[j]),        for j = 0, ..., n-1
#        x[j] free, y[j] >= 0,    for j = 0, ..., n-1
#  where pwl(x) = 0,     if x  = 0
#               = 1+|x|, if x != 0
#
#  Note
#   1. sum pwl(x[j]) <= b is to bound x vector and also to favor sparse x vector.
#      Here b = 3 means that at most two x[j] can be nonzero and if two, then
#      sum x[j] <= 1
#   2. pwl(x) jumps from 1 to 0 and from 0 to 1, if x moves from negative 0 to 0,
#      then to positive 0, so we need three points at x = 0. x has infinite bounds
#      on both sides, the piece defined with two points (-1, 2) and (0, 1) can
#      extend x to -infinite. Overall we can use five points (-1, 2), (0, 1),
#      (0, 0), (0, 1) and (1, 2) to define y = pwl(x)

import gurobipy as gp 
try:
    n=5  # 决策变量个数
    m=5   # 常系数约束条件矩阵行数
    c=[0.5,0.8,0.5,0.1,-1] # 目标函数系数矩阵
    A=[[0,0,0,1,-1],[0,0,1,1,-1],[1,1,0,0,-1],[1,0,1,0,-1],[1,0,0,1,-1]]# 常系数约束条件矩阵
    model=gp.Model('gc_pwl')

    # vars
    x=model.addVars(n,lb=-gp.GRB.INFINITY,name='x')
    y=model.addVars(n,name='y')   # y变量不会出现于目标函数

    # obj
    model.setObjective(gp.quicksum(c[j]*x[j] for j in range(n)),gp.GRB.MAXIMIZE)  # 求和写法

    # cons
    for i in range(m):
        model.addConstr(gp.quicksum(A[i][j]*x[j] for j in range(n))<=0)
    model.addConstr(y.sum()<=3)

    # PWL分段线性化约束，5个特征点[(-1, 2), (0, 1), (0, 0), (0, 1), (1, 2)]
    for j in range(n):
        model.addGenConstrPWL(x[j],y[j],[-1,0,0,0,1],[2,1,0,1,2])
    
    # opt
    model.optimize()

    for j in range(n):
        print('%s = %g'%(x[j].varName,x[j].x))
    print('Obj: %g'%(model.objVal))
except gp.GurobiError as e:
    print('Error code '+str(e.errno)+': '+str(e))
except AttributeError:
    print('Encountered an attribute error')

