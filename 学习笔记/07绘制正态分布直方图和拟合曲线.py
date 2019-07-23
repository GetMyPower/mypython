#������̬�ֲ�ֱ��ͼ���������
import random
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def demo1():
    #��python�л���̬�ֲ�ֱ��ͼ
    mu ,sigma = 0, 1
    sampleNo = 1000000
    np.random.seed(0)
    s = np.random.normal(mu, sigma, sampleNo)

    plt.hist(s, bins=100, normed=True)  #bin:�ܹ��м�����״ͼ
    plt.show()
def demo2():
    #��ֱ��ͼ����ʷֲ�����
    mu, sigma , num_bins = 0, 1, 50
    x = mu + sigma * np.random.randn(1000000)
    # ��̬�ֲ�������
    n, bins, patches = plt.hist(x, num_bins, normed=True, facecolor = 'blue', alpha = 0.5)

    # ֱ��ͼ������xΪx���ֵ��normed=1��ʾΪ�����ܶȣ�����Ϊһ����ɫ���飬ɫ�����0.5.����n�����ʣ�ֱ��������ߵ�xֵ���������������patches[50]
    y = mlab.normpdf(bins, mu, sigma)    #���һ�������̬�ֲ�����y

    plt.plot(bins, y, 'r--')
    plt.xlabel('Expectation')
    plt.ylabel('Probability')
    plt.title('histogram of normal distribution: $\mu = 0$, $\sigma=1$')

    plt.subplots_adjust(left = 0.15)  #������ͼ��Χ��߾�
    plt.show()

#demo1()
demo2()

