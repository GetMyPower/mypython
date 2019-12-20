# piecewise，
# 增加自定义的目标函数，
# 通过离散化获取(x,y)的对应集来实现优化
import gurobipy as gp
from math import exp


def f(u):
    return exp(u)


def g(u):
    return 2 * u * u - 4 * u


try:
    model = gp.Model()

    # vars
    lb, ub = 0, 1
    x = model.addVar(lb, ub, name='x')
    y = model.addVar(lb, ub, name='y')
    z = model.addVar(lb, ub, name='z')

    # obj
    model.setObjective(-y)  # ???
    npts = 101   # ???
    ptu = []  # u-自变量
    ptf = []  # f-函数
    ptg = []  # g-函数

    for i in range(npts):
        ptu.append(lb+(ub-lb)*i/(npts-1))  # 在[lb, ub]内线性地取npts-1个值
        ptf.append(f(ptu[i]))
        ptg.append(g(ptu[i]))

    model.setPWLObj(x, ptu, ptf)
    model.setPWLObj(z, ptu, ptg)

    # cons
    model.addConstr(x+2*y+3*z <= 4, 'c0')
    model.addConstr(x+y >= 1, 'c1')

    # opt
    model.optimize()

    print('IsMIP: %d' % (model.IsMIP))
    for v in model.getVars():
        print('%s %g' % (v.VarName, v.X))
    print('Obj: %g' % (model.ObjVal))

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')
