import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
from pylab import *
import pandas as pd

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 解决绘图中文字体乱码问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决绘图坐标的负号问题

filename = "rain.csv"
datafile = pd.read_csv(filename, header=None, prefix="V")  # 在没有列标题时，给列添加前缀。例如：添加‘X’ 成为 X0, X1, ...
summary = datafile.describe()
# print(summary)

# region 1 绘制箱型图
figure(1)
A = datafile.iloc[:, 1:13].values
boxplot(A)
plt.tick_params(labelsize=15)
plt.xlabel("month", size=15)
plt.ylabel("rain", size=15)
# endregion

# region 2 绘制10个月的降水趋势
figure(2)
minrings = -1
maxrings = 99
nrows = len(datafile)
for i in range(1, nrows):
    datarow = datafile.iloc[i, 1:13]
    labelcolor = (datafile.iloc[i, 12] - minrings) / (maxrings - minrings)
    datarow.plot(color=plt.cm.RdYlBu(labelcolor), alpha=0.5, )
plt.tick_params(labelsize=15)
plt.xlabel("month", size=15)
plt.ylabel("rain", size=15)

# endregion

# region 3 绘制相关矩阵
figure(3)
cormat=pd.DataFrame(datafile.iloc[1:len(datafile),1:13])
plt.pcolor(cormat)

# endregion


plt.show()
print("end")

# region

# endregion
