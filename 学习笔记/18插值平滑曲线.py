"""
用插值实现曲线的平滑化
https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.interpolate import interp1d
import matplotlib
# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 原始数据
NN = 500
x = np.arange(0, NN, 1)
y = 5*np.sin(x/50)+np.random.normal(0,0.1,NN)


# region 定义二次光滑曲线的函数
def get_smooth_quadratic(x,y,N):
    xx= np.linspace(x.min(), x.max(), N)
    f = interp1d(x, y, kind='quadratic')
    return xx,f(xx)
# endregion

plt.figure(1)
N=100    # 平滑处理后点数
xx,yy=get_smooth_quadratic(x,y,N)
plt.xticks(np.arange(0,500,60))
plt.title("二次光滑曲线")
plt.plot(xx,yy)
# plt.show()


# 各种插值方式的展示
plt.figure(2)
NN = 500
x = np.arange(0, NN, 1)
y = 5 * np.sin(x / 50) + np.random.normal(0, 0.1, NN)
xx = np.linspace(x.min(), x.max(), 100)
kindlist = ['linear', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next']
for n in kindlist:
    f = interp1d(x, y, kind=n)
    plt.plot(xx, f(xx), label=n)
plt.legend()
plt.show()
