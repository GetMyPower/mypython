"""
假设y=f(x)=0.3*x+0.15表达式未知，只知道是线性函数，
给定一组训练数据x_data和y_data，
使神经网络拟合模型y_model = weight * x_data + bias，
计算weight和bias，并绘制拟合后的图线
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.random.randn(10)
y_data = x_data * 0.3 + 0.15

weight = tf.Variable(0.5)
bias = tf.Variable(0.0)
y_model = weight * x_data + bias

loss = tf.pow((y_model - y_data),2)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for _ in range(200):
    sess.run(train_op)
    print(weight.eval(sess),bias.eval(sess))

plt.plot(x_data, y_data, 'ro', label='Original data')
plt.plot(x_data, sess.run(weight) * x_data + sess.run(bias), label='Fitted line')
plt.legend()
plt.show()
