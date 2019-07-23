# ��pulp����������Թ滮
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
print("���ֵ Z Ϊ", value(prob.objective), "����λ")
for v in prob.variables():
    print("����ֵ", v.name, ":", v.varValue, "��λ")