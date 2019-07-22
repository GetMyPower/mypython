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


plt.show()

# regioin

# endregion

print("end")
