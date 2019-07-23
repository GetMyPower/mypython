#绘制正态分布直方图和拟合曲线
import random
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def demo1():
    #在python中画正态分布直方图
    mu ,sigma = 0, 1
    sampleNo = 1000000
    np.random.seed(0)
    s = np.random.normal(mu, sigma, sampleNo)

    plt.hist(s, bins=100, normed=True)  #bin:总共有几条条状图
    plt.show()
def demo2():
    #画直方图与概率分布曲线
    mu, sigma , num_bins = 0, 1, 50
    x = mu + sigma * np.random.randn(1000000)
    # 正态分布的数据
    n, bins, patches = plt.hist(x, num_bins, normed=True, facecolor = 'blue', alpha = 0.5)

    # 直方图函数，x为x轴的值，normed=1表示为概率密度，即和为一，蓝色方块，色深参数0.5.返回n个概率，直方块左边线的x值，及各个方块对象patches[50]
    y = mlab.normpdf(bins, mu, sigma)    #拟合一条最佳正态分布曲线y

    plt.plot(bins, y, 'r--')
    plt.xlabel('Expectation')
    plt.ylabel('Probability')
    plt.title('histogram of normal distribution: $\mu = 0$, $\sigma=1$')

    plt.subplots_adjust(left = 0.15)  #整个绘图范围左边距
    plt.show()

#demo1()
demo2()

