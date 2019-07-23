import random
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import cholesky
from sympy import *


sampleNo = 10000;
# һά��̬�ֲ�
mu = 3
sigma = 0.1

#��̬�ֲ�
np.random.seed(0)
s = np.random.normal(mu, sigma, sampleNo )
plt.subplot(131)
plt.hist(s, 30, normed=True)

np.random.seed(0)
s = sigma * np.random.randn(sampleNo ) + mu
plt.subplot(132)
plt.hist(s, 30, normed=True)


#������̬�ֲ�
s=np.random.lognormal(3.7,0.92,sampleNo)
plt.subplot(133)
plt.hist(s, 100, normed=true)
plt.show()