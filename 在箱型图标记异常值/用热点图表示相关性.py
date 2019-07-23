import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
from pylab import *
import pandas as pd

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 解决绘图中文字体乱码问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决绘图坐标的负号问题

# data = np.mat([[1, 200, 105, 3, False], [2, 165, 80, 2, False], [3, 184.5, 120, 2, False], [4, 116, 70.8, 1, False],
#                [5, 270, 150, 4, True]])

filename = "dataTest.csv"
datafile = pd.read_csv(filename, header=None, prefix="V")  # 在没有列标题时，给列添加前缀。例如：添加‘X’ 成为 X0, X1, ...
summary = datafile.describe()

# region 3 箱型图标记异常值
figure(3)
# print(datafile.head())
# print(datafile.tail())
# summary=datafile.describe()
# print(summary)

# 取部分数据
array = datafile.iloc[:, 10:16].values
bp = boxplot(array)
plt.xlabel("属性", size=15)
plt.ylabel("分值", size=15)

# 获取异常值
x = [kk.get_xdata() for kk in bp['fliers']]
y = [kk.get_ydata() for kk in bp['fliers']]

# 标记异常值
for kk in range(len(x)):
    for i in range(len(x[kk])):
        plt.annotate(y[kk][i], xy=(x[kk][i], y[kk][i]), xytext=(x[kk][i] + 0.05, y[kk][i]))

print()

# endregion

# region 4 每线不同颜色
figure(4)
minRings = -1
maxRings = 99
nrows = 10
for i in range(nrows):
    datarow = datafile.iloc[i, 1:10]
    labelcolor = (datafile.iloc[i, 10] - minRings) / (maxRings - minRings) * 10  #
    datarow.plot(color=plt.cm.RdYlBu(labelcolor), alpha=0.5)
plt.xlabel("属性", size=15)
plt.ylabel("分值", size=15)

# endregion

# region 5 用热点图表示相关性
figure(5)
corMat = pd.DataFrame(datafile.iloc[1:20, 1:20].corr())  # corr用于计算列与列之间相关性
plt.pcolor(corMat, edgecolors='k', linewidths=1)

# 这里用于表示pcolor函数还可绘制numpy矩阵
# aaa = np.random.random((4, 5))
# plt.pcolor(aaa, edgecolors='k', linewidths=1)

# endregion

plt.show()
print("end")

# region

# endregion
