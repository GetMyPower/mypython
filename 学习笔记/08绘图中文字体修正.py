# 可以参照https://www.jianshu.com/p/4b3ec8b820c0，作者 蓝色的小妖精
import numpy as np


from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']   #解决绘图中文字体乱码问题
mpl.rcParams['axes.unicode_minus']=False       #解决绘图坐标的负号问题


plt.xlabel('电动汽车渗透率/%')
plt.xticks(np.arange(0,24,1))    #x刻度