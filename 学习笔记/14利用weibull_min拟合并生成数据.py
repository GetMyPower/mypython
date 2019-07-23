from scipy.stats import weibull_min
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 解决绘图中文字体乱码问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决绘图坐标的负号问题

# region 1.指定尺度参数和形状参数，生成一组威布尔分布的数据wbldata
scl = 2.89  # scale
shp = 1.95  # shape
wbldata = weibull_min.rvs(shp, loc=0, scale=scl, size=1000)

# endregion

# region 2.输入风速数据，拟合威布尔分布的两个参数
windspeed = [1, 0.86, 2.2, 7.06, 9.75, 10.57, 11.23, 10.23, 6.9, 4.19, 3.11, 2.86, 3.01, 2.89, 2.47, 2.48, 2.46, 1.96, 1.81,
      1.92, 2.18, 2.52, 2.34, 3.25, 4.46, 5.67, 7.17, 10.05, 10.62, 11.4, 11.45, 11, 9.39, 8.99, 8.9, 8.59, 8.7, 8.15,
      7.47, 6.78, 6.25, 6.25, 5.84, 5.58, 5.33, 5.74, 6.31, 6.55, 6.8, 7.21, 7.88, 7.47, 8.16, 8.73, 9.91, 10.11, 8.51,
      8.11, 7, 5.98, 5.07, 4.66, 4.16, 3.69, 3.3, 2.54, 1.89, 1.89, 1.95, 2.34, 2.04, 2.66, 3.3, 3.58, 3.87, 1.99, 2.1,
      2.99, 3.46, 3.26, 2.43, 1.78, 1.39, 0.93, 0.84, 1.14, 1.7, 1.7, 1.13, 1.34, 1.34, 1.49, 1.72, 1.79, 2.01, 1.34,
      0.79, 0.35, 0.93, 4.76, 5.47, 6.11, 6.31, 5.46, 4.7, 4.47, 4.19, 3.72, 3.26, 2.75, 2.28, 1.35, 0.88, 0.43, 0.89,
      1.14, 1.56, 2.01, 1.34, 1.14, 1.05, 2.21, 3.21, 5.9, 6.99, 6.77, 5.81, 4.48, 3.15, 2.02, 1.64, 1.54, 2.1, 2.29,
      2.48, 2.56, 2.43, 2.43, 2.28, 2.52, 3.02, 3.2, 2.88, 4.01, 5.07, 7.1, 9.31, 7.16, 8.7, 9.87, 10.48, 9.91, 8.12,
      6.79, 6.01, 5.42, 4.87, 4.48, 4.87, 5.1, 4.93, 4.61, 4.73, 4.32, 4.19, 3.96, 3.25, 4.36, 7.09, 10.11, 12.33, 12.1,
      13.34, 13.61, 12.52, 8.9, 6.2, 5.42, 4.85, 4.48, 3.55, 2.69, 2.34, 1.54, 1.5, 1.96, 2.19, 2.19, 2.47, 2.34, 2.12,
      2.08, 2.37, 2.18, 3.37, 5.43, 6.58, 7.35, 7.4, 6.46]

(shape, loc, scale) = weibull_min.fit(windspeed)   # 拟合所得数据的参数
print((shape, loc, scale))

plt.figure(1)
plt.subplot(1, 2, 1)
plt.title("原始风速数据直方图")
plt.hist(windspeed, density=True, histtype='stepfilled', alpha=0.2)
plt.subplot(1, 2, 2)
plt.title("原始风速数据")
plt.plot(np.arange(0, len(windspeed), 1), windspeed)
# endregion

# region 3.根据拟合得到参数，生成一组新的威布尔数据
windspeed_new = weibull_min.rvs(shape, loc=0, scale=scale, size=100)
plt.figure(2)
plt.subplot(1, 2, 1)
plt.title("拟合所得风速数据直方图")
plt.hist(windspeed_new, density=True, histtype='stepfilled', alpha=0.2)
plt.subplot(1, 2, 2)
plt.title("拟合所得风速数据")
plt.plot(np.arange(0, len(windspeed_new), 1), windspeed_new)
# endregion

plt.show()
