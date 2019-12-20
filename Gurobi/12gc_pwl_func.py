# gc_pwl_func.py

# This example considers the following nonconvex nonlinear problem
#
#  maximize    2 x    + y
#  subject to  exp(x) + 4 sqrt(y) <= 9
#              x, y >= 0
#
#  We show you two approaches to solve this:
#
#  1) Use a piecewise-linear approach to handle general function
#     constraints (such as exp and sqrt).
#     a) Add two variables
#        u = exp(x)
#        v = sqrt(y)
#     b) Compute points (x, u) of u = exp(x) for some step length (e.g., x
#        = 0, 1e-3, 2e-3, ..., xmax) and points (y, v) of v = sqrt(y) for
#        some step length (e.g., y = 0, 1e-3, 2e-3, ..., ymax). We need to
#        compute xmax and ymax (which is easy for this example, but this
#        does not hold in general).
#     c) Use the points to add two general constraints of type
#        piecewise-linear.
#
#  2) Use the Gurobis built-in general function constraints directly (EXP
#     and POW). Here, we do not need to compute the points and the maximal
#     possible values, which will be done internally by Gurobi.  In this
#     approach, we show how to "zoom in" on the optimal solution and
#     tighten tolerances to improve the solution quality.


import math
import gurobipy as gp


def printsol(model, x, y, u, v):
    print('x = '+str(x.x)+', u = '+str(u.x))
    print('y = '+str(y.x)+', v = '+str(v.x))
    print('Obj = '+str(model.objVal))

    vio = math.exp(x.x)+4*math.sqrt(y.x)-9   # 我们要求vio要<=0
    vio = 0 if vio < 0 else vio
    print('Vio = '+str(vio))


try:
    model = gp.Model()
    model.Params.OutPutFlag = 0   # 0静默输出
    x = model.addVar(name='x')
    y = model.addVar(name='y')
    u = model.addVar(name='u')
    v = model.addVar(name='v')

    model.setObjective(2*x+y, gp.GRB.MAXIMIZE)

    lc = model.addConstr(u+4*v <= 9)   # 以u和v表示的线性约束

    # region 方法1：手动创建u=exp(x)和v=sqrt(y)的PWL映射数据集
    xpts, ypts, upts, vpts = [], [], [], []
    intv = 1e-3

    # 创建u=exp(x)的分段线性映射
    xmax = math.log(9)
    t = 0
    while t < xmax+intv:
        xpts.append(t)
        upts.append(math.exp(t))
        t += intv

    # 创建v=sqrt(y)的分段线性映射
    ymax = (9/4)*(9/4)
    t = 0
    while t < ymax+intv:
        ypts.append(t)
        vpts.append(math.sqrt(t))
        t += intv

    gc1 = model.addGenConstrPWL(x, u, xpts, upts, 'gc1')  # 增加u=exp(x)的PWL约束
    gc2 = model.addGenConstrPWL(y, v, ypts, vpts, 'gc2')   # 增加v=sqrt(y)的PWL约束
    model.optimize()
    print('===================(1)===================')
    printsol(model, x, y, u, v)

    # endregion

    # region 方法2：根据指定的函数模型，Gurobi于内部自动创建u=exp(x)和v=sqrt(y)的PWL映射数据集
    # (2.1)
    model.reset()
    model.remove(gc1)
    model.remove(gc2)
    model.update()

    # u=exp(x)
    gcf1 = model.addGenConstrExp(x, u, name='gcf1')
    # v=sqrt(x)
    gcf2 = model.addGenConstrPow(y, v, 0.5, name='gcf2')

    model.params.FuncPieces = 1
    model.params.FuncPieceLength = 1e-3

    model.optimize()
    print('===================(2.1)===================')
    printsol(model, x, y, u, v)

    # (2.2)Zoom in, 在最优解的位置上，更精确地分段，计算更精确的解
    x.lb = max(x.lb, x.x-0.01)
    x.ub = min(x.ub, x.x+0.01)
    y.lb = max(y.lb, y.x-0.01)
    y.ub = min(y.ub, y.x+0.01)
    model.update()
    model.reset()
    model.params.FuncPieceLength = 1e-5

    model.optimize()
    print('===================(2.2)===================')
    printsol(model, x, y, u, v)

    # endregion
except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')
