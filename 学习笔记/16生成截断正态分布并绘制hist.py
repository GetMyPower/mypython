import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
from pylab import *

mu, sigma = 5, 0.7
lower, upper = mu - 2 * sigma, mu + 2 * sigma  # ½Ø¶ÏÔÚ[mu-2*sigma, mu+2*sigma]
X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
N = stats.norm(loc=mu, scale=sigma)


figure(1)
subplot(2,1,1)
plt.hist(X.rvs(10000), normed=True, bins=30)
subplot(2,1,2)
plt.hist(N.rvs(10000), normed=True, bins=30)
plt.show()