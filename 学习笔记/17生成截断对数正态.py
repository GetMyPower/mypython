"""
正态（0，1）---> exp() --->对数正态(0,1) ---> ln() ---> 正态(0,1)
对数正态的上下限[log_lower,log_upper]转为正态的上下限为np.log([log_lower,log_upper])
"""
import numpy as np
from pylab import *
from scipy import stats
# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

data1 = np.random.normal(0, 1, 10000)
data2 = exp(data1)
data3 = np.random.lognormal(0, 1, 10000)
data4 = np.log(data3)

figure(1)
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


# region 【功能函数】生成截断对数正态分布，要求对数正态在[log_lower,log_upper]
def get_trunc_lognorm(mu, sigma, log_lower, log_upper=np.inf, data_num=10000):
    norm_lower = np.log(log_lower)
    norm_upper = np.log(log_upper)
    X = stats.truncnorm((norm_lower - mu) / sigma, (norm_upper - mu) / sigma, loc=mu, scale=sigma)
    norm_data = X.rvs(data_num)
    log_data = np.exp(norm_data)
    return norm_data, log_data
# endregion

mu, sigma = 0, 1
norm_data, log_data = get_trunc_lognorm(mu, sigma, 5, 10)

figure(2)
subplot(2, 1, 1)
plt.hist(norm_data, normed=1, bins=30)
plt.xticks(np.arange(mu - 5 * sigma, mu + 5 * sigma, 0.5))
plt.title("中间过程的截断正态分布")

subplot(2, 1, 2)
plt.hist(log_data, normed=1, bins=30)
plt.xticks(np.arange(0, 50, 5))
# plt.xlim(0,50)
plt.title("所求的截断对数正态分布")
plt.show()

plt.show()
