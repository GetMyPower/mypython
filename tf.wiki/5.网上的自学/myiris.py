"""
目标：建立Dense分类模型对莺尾花分类。
重点：对字符label的整数编码encoder.fit_transform()、
onehot编码np_utils.to_categorical()，
以及对预测数据的argmax编码np.argmax()、
反整数编码encoder.inverse_transform()
"""
# !wget http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow.keras.utils as np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# region (1)读入原始数据，整数化标签，获取训练集测试集
dataframe = pd.read_csv("./Input/iris.data.csv", header=None)
dataset = dataframe.values  # (150, 5)
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y)  # 将文字编码为整数
dummy_y = np_utils.to_categorical(encoded_Y)  # 获取整数矩阵Y的one-hot矩阵，(150, 3)  [[1. 0. 0.], [1. 0. 0.], [1. 0. 0.], ……]
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3, random_state=0)  # 划分训练集测试集
print("1. Splited train and test dataset.")
# endregion

# region (2)建立模型并训练
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=4))
model.add(Dense(5, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("2. Model has been set up.")
model.summary()  # 打印网络结构

history = model.fit(X_train, Y_train, epochs=100 + 200, batch_size=8, validation_data=(X_test, Y_test),
                    verbose=2, shuffle=False)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Loss of training")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.subplot(1, 2, 2)
plt.title("Acc of training")
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
# endregion

# region (3)预测
pred = model.predict(X_test)  # 获得one-hot形式的预测数据，数据格式与dummy_y对应
pred = np.argmax(pred, axis=1)  # 将one-hot形式的pred数据转为整数编码数据，数据格式与encoded_Y对应
pred_label = encoder.inverse_transform(pred)  # 将整数编码的数据进行反编码获得字符串，数据格式与Y对应
Y_test_label = encoder.inverse_transform(np.argmax(Y_test, axis=1))
# print("test中实际标签为%s" % (Y_test_label))
# print("test中预测标签为%s" % (pred_label))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("comparison of train")
plt.plot(np.argmax(Y_train, axis=1), label='y_train')
plt.plot(np.argmax(model.predict(X_train), axis=1), label='Y_hat_train')
plt.yticks(np.arange(3))
plt.legend()
plt.subplot(1, 2, 2)
plt.title("comparison of test")
plt.plot(np.argmax(Y_test, axis=1), label='y_test')
plt.plot(np.argmax(model.predict(X_test), axis=1), label='Y_hat_test')
plt.yticks(np.arange(3))
plt.legend()

plt.show()
# endregion
