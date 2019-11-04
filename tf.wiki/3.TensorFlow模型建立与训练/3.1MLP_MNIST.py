# coding=gbk
""""
采用多层感知机MLP搭建深度学习，
训练和测试MNIST数据集
"""
import time

start = time.clock()
import tensorflow as tf
# tf.enable_eager_execution()   # 在colab中有用
import numpy as np

# region 1.数据载入
class MNISTLoader():
    def __init__(self):
        mnist=tf.keras.datasets.mnist
        (self.train_data,self.train_label),(self.test_data,self.test_label)=mnist.load_data()
        self.train_data = np.expand_dims(self.train_data.astype(np.float32)/255.0,axis = -1)
        self.test_data = np.expand_dims(self.test_data.astype(np.float32)/255.0,axis=-1)
        self.train_label = self.train_label.astype(np.int32)
        self.test_label = self.test_label.astype(np.int32)
        self.num_train_data,self.num_test_data = self.train_data.shape[0],self.test_data.shape[0]

    def get_batch(self,batch_size):
        index = np.random.randint(0,np.shape(self.train_data)[0],batch_size)
        return self.train_data[index,:],self.train_label[index]



# endregion

# region 2.MLP模型构建
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten=tf.keras.layers.Flatten()
        self.dense1=tf.keras.layers.Dense(units=100,activation=tf.nn.relu)
        self.dense2=tf.keras.layers.Dense(units=10)
    def call(self,inputs):
        x=self.flatten(inputs)
        x=self.dense1(x)
        x=self.dense2(x)
        output=tf.nn.softmax(x)
        return output

# endregion

# region 3.模型训练
num_epochs=5
batch_size=50
learning_rate=0.001

model=MLP()
data_loader=MNISTLoader()
optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)

num_batches=int(data_loader.num_train_data//batch_size*num_epochs)
for batch_index in range(num_batches):
    X,y=data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_pred=model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y,y_pred=y_pred)
        loss=tf.reduce_mean(loss)
        if (batch_index+1)%400==0:
            print("batch %d, loss %f"%(batch_index,loss.numpy()))
    grads=tape.gradient(loss,model.variables)
    optimizer.apply_gradients(zip(grads,model.variables))



# endregion

# region 4.模型评估
sparse_categorical_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()
num_batches=int(data_loader.num_test_data//batch_size)
for batch_index in range(num_batches):
    start_index,end_index=batch_index*batch_size,(batch_index+1)*batch_size
    y_pred=model.predict(data_loader.test_data[start_index:end_index])
    sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index:end_index],y_pred=y_pred)
print("test accuracy %f"%sparse_categorical_accuracy.result())

# endregion

end = time.clock()
print("\n运行时长 %.3f Seconds" % (end - start))

# region

# endregion
