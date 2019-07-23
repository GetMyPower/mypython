import numpy as np
from scipy import stats


aaa=np.random.normal(1,3,1000)
mu,sigma = stats.norm.fit(aaa)
MMM=np.random.normal(mu,sigma,15000)