# 用pulp求解整数线性规划
from pulp import *

prob = LpProblem("problem1", LpMaximize)
x1 = LpVariable("x1", 0, None, LpInteger)
x2 = LpVariable("x2", 0, None, LpInteger)
prob += x1 - x2
prob += x1 + x2 <= 2.5
prob += x1 + 0 * x2 >= 0
prob += 0 * x1 + x2 >= 0
prob.writeLP("problem1.lp")
prob.solve()
print("最大值 Z 为", value(prob.objective), "个单位")
for v in prob.variables():
    print("最优值", v.name, ":", v.varValue, "单位")