# coding=gbk
""""
神经网络拟合正弦函数

"""
import time

start = time.clock()
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

learning_rate = 0.001
batch_size = 100
num_epochs = 200


# region 1.数据载入
class DataLoader():
    def my_func(self, x):
        return np.sin(x)

    def __init__(self):
        train_data_raw = np.arange(0, 2 * np.pi, 0.002)
        train_label_raw = self.my_func(train_data_raw)

        test_data_raw = np.random.uniform(0, 2 * np.pi, 100)
        test_label_raw = self.my_func(test_data_raw)

        self.data_bound = (min(train_data_raw), max(train_data_raw))
        self.label_bound = (min(train_label_raw), max(train_label_raw))

        # 归一化
        self.train_data = (train_data_raw - self.data_bound[0]) / (self.data_bound[1] - self.data_bound[0])
        self.train_label = (train_label_raw - self.label_bound[0]) / (self.label_bound[1] - self.label_bound[0])
        self.test_data = (test_data_raw - self.data_bound[0]) / (self.data_bound[1] - self.data_bound[0])
        self.test_label = (test_label_raw - self.label_bound[0]) / (self.label_bound[1] - self.label_bound[0])
        self.num_train_data, self.num_test_data = len(self.train_data), len(self.test_data)

    def get_batch(self, batch_size):
        index = np.random.randint(0, self.num_train_data, batch_size)
        return self.train_data[index], self.train_label[index]


# endregion
data_loader = DataLoader()


# region 2.建立MLP模型
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=10, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10, activation=tf.nn.relu)
        self.dense2_add1 = tf.keras.layers.Dense(units=5, activation=tf.nn.relu)
        self.dense2_add2 = tf.keras.layers.Dense(units=5, activation=tf.nn.relu)
        self.dense2_add3 = tf.keras.layers.Dense(units=5, activation=tf.nn.relu)
        self.dense2_add4 = tf.keras.layers.Dense(units=5 + 2, activation=tf.nn.relu)
        self.dense2_add5 = tf.keras.layers.Dense(units=5 + 2, activation=tf.nn.relu)
        self.dense2_add6 = tf.keras.layers.Dense(units=5, activation=tf.nn.relu)
        self.dense3_out = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense2_add1(x)
        x = self.dense2_add2(x)
        x = self.dense2_add3(x)
        # x = self.dense2_add4(x)
        # x = self.dense2_add5(x)
        x = self.dense3_out(x)
        output = x
        return output


# endregion

# region 3.训练模型
def train():
    model = MLP()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    num_batches = int(data_loader.num_train_data // batch_size * num_epochs)

    checkpoint = tf.train.Checkpoint(myAwesomeModel=model)
    manager = tf.train.CheckpointManager(checkpoint, directory='./save',
                                         max_to_keep=1)  # 使用tf.train.CheckpointManager管理Checkpoint

    for batch_index in range(1, num_batches + 1):
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.multiply(tf.reduce_sum(tf.pow(tf.squeeze(y_pred) - y, 2)), 100)
            if batch_index % 500 == 0:
                print("batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))

        if batch_index % 1000 == 0 or batch_index == num_batches:
            path = manager.save(checkpoint_number=batch_index)  # (2)这里用checkpoint_manager保存模型并编号
            print("model saved to %s" % path)


# endregion

# region 4.测试
def test():
    my_model = MLP()
    checkpoint = tf.train.Checkpoint(myAwesomeModel=my_model)
    checkpoint.restore(tf.train.latest_checkpoint('./save'))

    # 总误差计算
    y_pred = my_model.predict(data_loader.test_data)
    y_true = data_loader.test_label
    loss = tf.multiply(tf.reduce_sum(tf.pow(tf.squeeze(y_pred) - y_true, 2)), 100)
    print("loss = %f" % (loss))

    # 抽取几个数，归一化之后预测，并还原
    X_raw = np.arange(0, 2 * np.pi, 0.5)
    X_ = (X_raw - data_loader.data_bound[0]) / (data_loader.data_bound[1] - data_loader.data_bound[0])
    y_ = my_model.predict(X_)
    y_raw = y_ * (data_loader.label_bound[1] - data_loader.label_bound[0]) + data_loader.label_bound[0]

    y_true_raw = data_loader.my_func(X_raw)
    y_true = data_loader.my_func(X_)

    print("实际预测效果：\nX_raw\t\ty_raw\t\ty_true_raw")
    for (x, y, z) in zip(X_raw, y_raw, y_true_raw):
        print("%.4f\t\t%.4f\t\t%.4f" % (x, y, z))

    # print("归一化预测效果：\nX_\t\ty_\t\ty_true")
    # for (x, y, z) in zip(X_, y_, y_true):
    #     print("%.4f\t%.4f\t%.4f" % (x, y, z))


# endregion

train()
test()

end = time.clock()
print("\n运行时长 %.3f Seconds" % (end - start))

# region

# endregion
