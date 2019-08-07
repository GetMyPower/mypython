"""
正态（0，1）---> exp() --->对数正态(0,1) ---> ln() ---> 正态(0,1)
"""
import numpy as np
from pylab import *
# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

data1 = np.random.normal(0, 1, 10000)
data2 = exp(data1)
data3 = np.random.lognormal(0, 1, 10000)
data4 = np.log(data3)

figure(3)
subplot(2, 2, 1)
plt.hist(data1, normed=1, bins=30)
plt.title("正态(0,1)")
subplot(2, 2, 2)
plt.hist(data2, normed=1, bins=30)
plt.title("正态(0,1)取指数变化")
subplot(2, 2, 3)
plt.hist(data3, normed=1, bins=30)
plt.title("对数正态(0,1)")
subplot(2, 2, 4)
plt.hist(data4, normed=1, bins=30)
plt.title("对数正态(0,1)取ln()变化")

plt.show()
