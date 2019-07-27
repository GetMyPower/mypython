import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 解决绘图中文字体乱码问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决绘图坐标的负号问题



threshold = 1.0e-5
# x1_data = np.random.randn(100).astype(np.float32)
# x2_data = np.random.randn(100).astype(np.float32)
x1_data = np.linspace(-1, 1, 100)
x2_data = np.linspace(-1, 1, 100)
y_data = 2 * x1_data + 3 * x2_data + 1.5

weight1 = tf.Variable(0.5)
weight2 = tf.Variable(0.5)
bias = tf.Variable(1.0)
x1_ = tf.placeholder(tf.float32)
x2_ = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)
y_model = tf.add(tf.multiply(weight1, x1_), tf.add(tf.multiply(weight2, x2_), bias))

loss = tf.reduce_mean(tf.pow(y_model - y_, 2))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

flag = True
while flag:
    for (x, y) in zip(zip(x1_data, x2_data), y_data):
        sess.run(train_op, feed_dict={x1_: x[0], x2_: x[1], y_: y})
        print(
            "weight1 = %.4f | weight2 = %.4f | bias = %.4f" % (weight1.eval(sess), weight2.eval(sess), bias.eval(sess)))
        if sess.run(loss, feed_dict={x1_: x1_data, x2_: x2_data, y_: y_data}) < threshold:
            flag = False

figure(1)
ax = Axes3D(figure(1))
X, Y = np.meshgrid(x1_data, x2_data)
Z = sess.run(weight1) * X + sess.run(weight2) * Y + sess.run(bias)
ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=plt.get_cmap('rainbow'))  # rstride 和 cstride 分别代表 row 和 column 的跨度
ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=plt.get_cmap('seismic'))  # 绘制z=0的等高线
# ax.set_zlim(-8, 8)
ax.set_xlabel('x1', color='r', fontsize=15)  # 给三个坐标轴注明
ax.set_ylabel('x2', color='g', fontsize=15)
ax.set_zlabel('y=w1*x1 + w2*x2 + bias', color='b', fontsize=15)  # 给三个坐标轴注明

plt.tick_params(labelsize=15)
plt.show()
