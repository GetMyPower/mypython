# 备份Gurobi官方的部分例程

|文件|主题|重点|附加项|  
|:----:|:----:|:----|:----|  
|02qp.py|二次规划|(1) x.vType更改变量类型<br>(2) lb和ub|model._Model__vars可获取模型中决策变量的list|
|06dense.py|系数为全矩阵|(1) gp.QuadExpr()构建含二次项的目标函数<br>(2) 二维A矩阵写入约束的思想|model.write('dense.lp')输出模型到文件|
|08matrix1.py|系数为稀疏矩阵|(1) 采用scipy.sparse.csr_matrix()构建稀疏矩阵<br>(2) A @ x <= rhs构建目标函数和约束的语法形式||
|10piecewise.py|PWL分段线性化目标函数|model.setPWLObj(x, ptu, ptf)<br>目标函数中增加一项自定义函数||
|11gc_pwl.py|给定点集实现PWL约束|model.addGenConstrPWL(x[j],y[j],[-1,0,0,0,1],[2,1,0,1,2])<br>采用5个点集构建PWL|x要从小到大排序|
|12gc_pwl_func.py|内置语法增加指数约束|model.addGenConstrExp(x, u)<br>为模型增加一项指数约束|model.addConstr(u+4*v <= 9)<br>要首先构建用u和v表示的线性约束，再用addGenConstrExp构建x与u的关系|
|||||
