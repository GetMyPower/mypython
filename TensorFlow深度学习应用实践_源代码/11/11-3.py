"""
假设y=f(x)=0.3*x+0.15表达式未知，只知道是线性函数，
给定一组训练数据x_data和y_data，使神经网络拟合模型y_model = weight * x_data + bias，
计算weight和bias，并绘制拟合后的图线。
对比程序11-1.py，当x_data数据量较大时，
网络参数无法收敛，这时应当改变训练参数为一个一个输入。
对比程序11-2.py，将误差值作为迭代判据，改进loss的计算公式，
并且改变了y_model的表达式。
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

threshold = 1.0e-2
x_data = np.random.randn(100).astype(np.float32)
y_data = x_data * 3 + 1

weight = tf.Variable(1.)
bias = tf.Variable(1.)
x_ = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)
y_model = tf.add(tf.mul(x_, weight), bias)

loss = tf.reduce_mean(tf.pow((y_model - y_),2))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
flag = 1
while(flag):

    for (x,y) in zip(x_data,y_data):
        sess.run(train_op,feed_dict={x_:x,y_:y})
print(weight.eval(sess), bias.eval(sess))

    if sess.run(loss,feed_dict={x_:x_data,y_:y_data}) <= threshold:
        flag = 0

plt.plot(x_data, y_data, 'ro', label='Original data')
plt.plot(x_data, sess.run(weight) * (x_data) + sess.run(bias), label='Fitted line')
plt.legend()
plt.show()
