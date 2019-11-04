# coding=gbk
""""
һ��ʶ��MNIST�ļ�MLPģ�ͣ������checkpoint��ȡMLPģ�ͣ�
ʹ��tensorboard��ѵ�����̿��ӻ���
���ӻ�Ҫ����������ִ��tensorboard --logdir=./tensorboard������Ӧ��ҳ
TensorFlow �ṩ�� tf.train.Checkpoint ��һǿ��ı���������ָ��࣬
����ʹ���� save() �� restore() ����
�� TensorFlow �����а��� Checkpointable State �Ķ�����б���ͻָ���
������ԣ�tf.keras.optimizer �� tf.Variable �� tf.keras.Layer
���� tf.keras.Model ʵ�������Ա����档
"""
import time

start = time.clock()
import tensorflow as tf
import numpy as np
import argparse


# region 1.��������
class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST�е�ͼ��Ĭ��Ϊuint8��0-255�����֣������´��뽫���һ����0-1֮��ĸ������������������һά��Ϊ��ɫͨ��
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)  # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)  # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)  # [60000]
        self.test_label = self.test_label.astype(np.int32)  # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # �����ݼ������ȡ��batch_size��Ԫ�ز�����
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_label[index]


# endregion

# region 2.����MLPģ��
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()  # Flatten�㽫����һά��batch_size�������ά��չƽ
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):  # [batch_size, 28, 28, 1]
        x = self.flatten(inputs)  # [batch_size, 784]
        x = self.dense1(x)  # [batch_size, 100]
        x = self.dense2(x)  # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


# endregion

# region 3.�����в���
parse = argparse.ArgumentParser(description='process some integers')
parse.add_argument('--mode', default='train', help='train or test')
parse.add_argument('--num_epochs', default=1)
parse.add_argument('--batch_size', default=50)
parse.add_argument('--learning_rate', default=0.001)
args = parse.parse_args()
data_loader = MNISTLoader()


# endregion

# region 4.ѵ��ģ��
def train():
    model = MLP()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    num_batches = int(data_loader.num_train_data // args.batch_size * args.num_epochs)
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model)
    manager = tf.train.CheckpointManager(checkpoint, directory='./save',
                                         max_to_keep=3)  # ʹ��tf.train.CheckpointManager����Checkpoint
    summary_writer = tf.summary.create_file_writer('./tensorboard')   #ʵ������¼��
    for batch_index in range(1, num_batches + 1):
        X, y = data_loader.get_batch(args.batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            if batch_index % 100 == 0:
                print("batch %d: loss %f" % (batch_index, loss.numpy()))

            with summary_writer.as_default():  # ָ����¼��
                tf.summary.scalar("loss", loss, step=batch_index)  # ����ǰ��ʧ������ֵд���¼��
                

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))
        if batch_index % 200 == 0:
            # path = checkpoint.save('./save/model.ckpt')  # (1)������checkpointֱ�ӱ���ģ�͵ķ�ʽ
            path = manager.save(checkpoint_number=batch_index)  # (2)������checkpoint_manager����ģ�Ͳ����
            print("model saved to %s" % path)


# endregion

# region 5.����ģ��
def test():
    model_to_be_restored = MLP()
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model_to_be_restored)
    checkpoint.restore(tf.train.latest_checkpoint('./save'))
    y_pred = np.argmax(model_to_be_restored.predict(data_loader.test_data), axis=-1)
    print("test accuracy: %f" % (sum(y_pred == data_loader.test_label) / data_loader.num_test_data))


# endregion

if __name__ == '__main__':
    if args.mode == 'train':
        train()
    if args.mode == 'test':
        test()

end = time.clock()
print("\n����ʱ�� %.3f Seconds" % (end - start))

# region

# endregion
