import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
from pylab import *
import pandas as pd
import tensorflow as tf

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 解决绘图中文字体乱码问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决绘图坐标的负号问题

# 9.3节无法计算，不知道问题在哪
xs = np.random.randint(46, 99, 100)
ys = 1.7 * xs

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable(0.1)
b = tf.Variable(0.1)

y_ = tf.multiply(w, x) + b

loss = tf.reduce_sum(tf.pow(y - y_, 2))

train = tf.train.GradientDescentOptimizer(0.02).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(100):
    sess.run(train, feed_dict={x: xs, y: ys})
print("w:")
print(w.eval(sess))
print("b:")
print(b.eval(sess))
print(sess.run(y_, feed_dict={x: 50}))
