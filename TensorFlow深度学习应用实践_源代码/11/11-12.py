"""
loss太大，神经网络不收敛，偶尔无法计算。注意学习速率的设定
"""

import tensorflow as tf
import numpy as np

houses = 100
features = 2

x_data = np.zeros([houses, 2])
for i in range(houses):
    x_data[i, 0] = np.round(np.random.uniform(50., 150.))
    x_data[i, 1] = np.round(np.random.uniform(3., 7.))
weights = np.array([[2.], [3.]])
y_data = np.dot(x_data, weights)

x_data_ = tf.placeholder(tf.float32, [None, 2])
y_data_ = tf.placeholder(tf.float32, [None, 1])
weights_ = tf.Variable(np.ones([2, 1]), dtype=tf.float32)
y_model = tf.matmul(x_data_, weights_)

loss = tf.reduce_mean(tf.pow((y_model - y_data_), 2))

# 学习速率的设置要多加小心
train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for _ in range(30):
    for x, y in zip(x_data, y_data):
        z1 = x.reshape(1, 2)
        z2 = y.reshape(1, 1)
        sess.run(train_op, feed_dict={x_data_: z1, y_data_: z2})
    print("weights = ")
    print(weights_.eval(sess))
    print("loss = ")
    print(sess.run(loss, feed_dict={x_data_: x_data, y_data_: y_data}))
    print("------")
tf.reset_default_graph()
