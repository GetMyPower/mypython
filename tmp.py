# coding=gbk
import time

start = time.clock()
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

# region 
class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)
        self.train_label = self.train_label.astype(np.int32)
        self.test_label = self.test_label.astype(np.int32)
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_label.shape[0]

    def get_batch(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_label[index]


# endregion

# region 
class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)

        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)

        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        output = tf.nn.softmax(x)
        return output


# endregion

# region 
num_epochs=5   # 训练过程会使每个batch使用num_epochs次
bat_size=50   # 每个batch的大小
learning_rate=0.001

model=CNN()
data_loader=MNISTLoader()
optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)

num_batches=int(data_loader.num_train_data//bat_size*num_epochs)   # 一共有num_batches个允许重复的batch放入模型训练
for batch_index in range(num_batches):
    X,y=data_loader.get_batch(bat_size)
    with tf.GradientTape() as tape:
        y_pred=model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y,y_pred=y_pred)
        loss=tf.reduce_mean(loss)
        if batch_index%400==0:
            print("batch: %d, loss %f"%(batch_index,loss.numpy()))
    grads=tape.gradient(loss,model.variables)
    optimizer.apply_gradients(zip(grads,model.variables))


# endregion

# region 
sparse_categorical_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()
num_batches=int(data_loader.num_test_data//bat_size)
for batch_index in range(num_batches):
    start_index,end_index=batch_index*bat_size,(batch_index+1)*bat_size
    y_pred=model.predict(data_loader.test_data[start_index:end_index])
    sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index:end_index],y_pred=y_pred)
print("test accuracy: %f"%(sparse_categorical_accuracy.result()))

# endregion




end = time.clock()
print("\nrun time: %.4f Seconds" % (end - start))


# region

# endregion
