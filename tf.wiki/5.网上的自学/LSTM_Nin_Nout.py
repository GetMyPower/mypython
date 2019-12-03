"""
基于Keras的LSTM多变量时间序列预测https://blog.csdn.net/qq_28031525/article/details/79046718
数据文件https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv
目标：利用前N_in个时刻的pollution信息，预测下N_out个时刻的pollution
笔记：
1.将时序数据转化为有监督数据的方法，这里采用随机分离的方式获取train和test
2.回归问题，loss='mae', optimizer='adam'，评估采用RMSE
3.采用tf.keras.models以及fit来训练
4.sklearn.preprocessing.MinMaxScaler的归一化与反归一化
5.LabelEncoder对字符型变量的整数编码
6.DataFrame存储数据的方式，以及命名columns、删除列、提取元素
7.改变epochs和batch_size提升训练效率
8.使用tf.keras.models.save和tf.keras.models.load_model来存贮和读取HDF5格式的网络模型
9.plot(figsize=(5,5)) # 500*500的图，同样适用svg
"""

# region import
import time

start = time.perf_counter()
from datetime import datetime
from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import os
import numpy as np

# endregion

file_dir = "./Output"
if not os.path.isdir(file_dir):
    os.makedirs(file_dir)
file_dir = "./Input"
if not os.path.isdir(file_dir):
    os.makedirs(file_dir)

N_in = 24
N_out = 3


# region 1.读入数据并获取train和test

# 将时序数据集转化为有监督学习问题
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    for i in range(n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def get_train_and_test_data():
    # 读取最原始的数据并处理时间列
    parse = lambda x: datetime.strptime(x, '%Y %m %d %H')
    dataset = read_csv('./Input/PRSA_data_2010.1.1-2014.12.31.csv', parse_dates=[['year', 'month', 'day', 'hour']],
                       index_col=0, date_parser=parse)
    dataset.drop('No', axis=1, inplace=True)
    dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'
    dataset.drop(['dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain'], axis=1, inplace=True)  # 只保留一列数据
    dataset['pollution'].fillna(0, inplace=True)
    dataset = dataset[24:]
    dataset.to_csv('./Input/pollution.csv')

    # 归一化
    values = dataset.values
    values = values.astype('float32')
    scaled = scaler.fit_transform(values)

    # 绘制信息
    # groups = [0]  # 选择要绘制的列的索引
    # i = 1
    # plt.figure(figsize=(10, 10))
    # for group in groups:
    #     plt.subplot(len(groups), 1, i)
    #     plt.plot(values[:, group])
    #     plt.title(dataset.columns[group])
    #     i += 1

    # 时序转为有监督
    reframed = series_to_supervised(scaled, N_in, N_out)

    # 【方式1】前n_train_hours个数据为train，剩余的是test
    values = reframed.values
    n_train_hours = 365 * 24 * 3
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    train_X, train_y = train[:, :-N_out], train[:, -N_out:]
    test_X, test_y = test[:, :-N_out], test[:, -N_out:]
    train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1])
    test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])

    # 【方式2】随机采集n_train_hours个数据为train，剩余的是test
    values = reframed.values
    n_train_hours = 365 * 24 * 3
    ind_train = np.random.randint(0, len(values), n_train_hours)
    ind_test = np.delete(np.arange(len(values)), ind_train)
    train = values[ind_train]
    test = values[ind_test]
    train_X, train_y = train[:, :-N_out], train[:, -N_out:]
    test_X, test_y = test[:, :-N_out], test[:, -N_out:]
    train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1])
    test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])

    return train_X, train_y, test_X, test_y


# endregion
scaler = MinMaxScaler(feature_range=(0, 1))  # 归一化与反归一化的函数
train_X, train_y, test_X, test_y = get_train_and_test_data()


# region 2.设计模型并使用train和test进行fit
def disign_fit():
    # design network
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))

    # model.add(tf.keras.layers.Dense(10))  # 可以加一层

    model.add(tf.keras.layers.Dense(N_out))
    model.compile(loss='mae', optimizer='adam')
    history = model.fit(train_X, train_y, epochs=300, batch_size=10000, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    plt.figure(figsize=(7, 7))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig('./Output/history.svg')
    model.save('./Output/my_model.h5')


# endregion
disign_fit()


# region 3.评估模型
def eval(train_X, train_y, test_X, test_y):
    model = tf.keras.models.load_model('./Output/my_model.h5')  # 从文件读取模型
    # 预测值yhat与inv_yhat
    yhat = model.predict(test_X)
    inv_yhat = scaler.inverse_transform(yhat)

    # 实际值test_y与inv_y
    test_y = test_y.reshape(test_y.shape[0], N_out)
    inv_y = scaler.inverse_transform(test_y)

    # 方均根误差rmse
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('test RMSE: %.3f' % rmse)

    # 随便取100组数据绘图对比预测值与实际值
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    ind = np.random.randint(0, len(test_y), 100)  # 随便取100组数据绘图
    plt.plot(inv_yhat[ind].flatten(), label='yhat')
    plt.plot(inv_y[ind].flatten(), label='y')
    plt.legend()
    plt.subplot(1, 2, 2)
    ind = np.random.randint(0, len(test_y), 200)
    plt.hist(inv_yhat[ind].flatten(), bins=40, label='yhat')
    plt.hist(inv_y[ind].flatten(), bins=40, label='y')
    plt.legend()
    plt.savefig('./Output/y_vs_yhat.svg')


# endregion
eval(train_X, train_y, test_X, test_y)

plt.show()
end = time.perf_counter()
print("\nRunning time %d:%d:%.1f" % ((end - start) // 3600, ((end - start) % 3600) // 60, ((end - start) % 3600) % 60))
print("=============== Run to end successfully! ===============")

# region

# endregion
