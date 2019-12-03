""""
每15分钟的负荷数据文件地址：https://data.open-power-system-data.org/time_series/2019-06-05/time_series_15min_singleindex.csv
目标：建立LSTM预测模型，根据前N_in个负荷点的数据预测后N_out个负荷点的数据
disign_fit中提供了深度模型和LSTM模型，分别运行后可以发现LSTM比深度模型效率高、速度快
"""

# region import
import time

start = time.perf_counter()  # 程序开始运行时间
import tensorflow as tf
import numpy as np
import os
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

# endregion

file_dir = "./Output"
if not os.path.isdir(file_dir):
    os.makedirs(file_dir)
file_dir = "./Input"
if not os.path.isdir(file_dir):
    os.makedirs(file_dir)

N_in = 4 * 24 * 7  # 利用前7天的数据，预测后1天的数据。每天是4*24条数据
N_out = 4 * 24
k_train = 0.6  # 所有数据中，训练集比例为k_train，测试集比例为(1-k_train)
epochs = 300
batch_size = 10000   # LSTM可以对应比较大的batch_size，Dense则比较小

Model_type = 'LSTM'  # 选择模型结构，深度神经网络或者LSTM。'Dense' or 'LSTM'


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
    parse = lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ')  # 用于转换日期时间格式的函数
    dataset = read_csv('./Input/time_series_15min_singleindex.csv', parse_dates=['utc_timestamp'], index_col=0,
                       date_parser=parse)  # 'utc_timestamp'列将通过parse函数转换格式，并放回'utc_timestamp'列。index_col=0是将utc_timestamp作为索引列
    column_names = ["cet_cest_timestamp", "AT_load_actual_entsoe_transparency",
                    "AT_load_forecast_entsoe_transparency", "AT_price_day_ahead", "AT_solar_generation_actual",
                    "AT_wind_onshore_generation_actual", "BE_load_actual_entsoe_transparency",
                    "BE_load_forecast_entsoe_transparency", "BE_wind_onshore_capacity",
                    "BE_wind_onshore_generation_actual", "BE_wind_onshore_profile",
                    "DE_load_actual_entsoe_transparency", "DE_load_forecast_entsoe_transparency",
                    "DE_solar_capacity", "DE_solar_generation_actual", "DE_solar_profile", "DE_wind_capacity",
                    "DE_wind_generation_actual", "DE_wind_profile", "DE_wind_offshore_capacity",
                    "DE_wind_offshore_generation_actual", "DE_wind_offshore_profile", "DE_wind_onshore_capacity",
                    "DE_wind_onshore_generation_actual", "DE_wind_onshore_profile",
                    "DE_50hertz_load_actual_entsoe_transparency", "DE_50hertz_load_forecast_entsoe_transparency",
                    "DE_50hertz_solar_generation_actual", "DE_50hertz_solar_generation_forecast",
                    "DE_50hertz_wind_generation_actual", "DE_50hertz_wind_generation_forecast",
                    "DE_50hertz_wind_offshore_generation_actual", "DE_50hertz_wind_onshore_generation_actual",
                    "DE_50hertz_wind_onshore_generation_forecast", "DE_AT_LU_load_actual_entsoe_transparency",
                    "DE_AT_LU_load_forecast_entsoe_transparency", "DE_AT_LU_solar_generation_actual",
                    "DE_AT_LU_wind_offshore_generation_actual", "DE_AT_LU_wind_onshore_generation_actual",
                    "DE_LU_load_actual_entsoe_transparency", "DE_LU_load_forecast_entsoe_transparency",
                    "DE_LU_solar_generation_actual", "DE_LU_wind_offshore_generation_actual",
                    "DE_LU_wind_onshore_generation_actual", "DE_amprion_load_actual_entsoe_transparency",
                    "DE_amprion_load_forecast_entsoe_transparency", "DE_amprion_solar_generation_actual",
                    "DE_amprion_solar_generation_forecast", "DE_amprion_wind_generation_actual",
                    "DE_amprion_wind_generation_forecast", "DE_amprion_wind_onshore_generation_actual",
                    "DE_tennet_load_actual_entsoe_transparency", "DE_tennet_load_forecast_entsoe_transparency",
                    "DE_tennet_solar_generation_actual", "DE_tennet_solar_generation_forecast",
                    "DE_tennet_wind_generation_actual", "DE_tennet_wind_generation_forecast",
                    "DE_tennet_wind_offshore_generation_actual", "DE_tennet_wind_onshore_generation_actual",
                    "DE_transnetbw_load_actual_entsoe_transparency",
                    "DE_transnetbw_load_forecast_entsoe_transparency", "DE_transnetbw_solar_generation_actual",
                    "DE_transnetbw_solar_generation_forecast", "DE_transnetbw_wind_generation_actual",
                    "DE_transnetbw_wind_generation_forecast", "DE_transnetbw_wind_onshore_generation_actual",
                    "HU_load_actual_entsoe_transparency", "HU_load_forecast_entsoe_transparency",
                    "HU_wind_onshore_generation_actual", "LU_load_actual_entsoe_transparency",
                    "LU_load_forecast_entsoe_transparency", "NL_load_actual_entsoe_transparency",
                    "NL_load_forecast_entsoe_transparency", "NL_solar_generation_actual",
                    "NL_wind_offshore_generation_actual", "NL_wind_onshore_generation_actual"]  # 不包含utc_timestamp的各列名称
    column_names.remove("AT_load_actual_entsoe_transparency")  # AT_load_actual_entsoe_transparency是要保留的列
    dataset.drop(column_names, axis=1, inplace=True)  # 删除AT_load_actual_entsoe_transparency以外的列
    dataset.dropna(axis=0, inplace=True)  # 删除负荷值为nan的行
    dataset.index.name = 'date_time'  # 为日期时间列改名
    dataset.columns = ['load']  # 为负荷列AT_load_actual_entsoe_transparency改名
    dataset.to_csv('./Input/load_refresh.csv')
    del dataset

    # 读取处理过的负荷数据
    dataset = read_csv('./Input/load_refresh.csv', index_col=0)

    # 归一化
    values = dataset.values
    values = values.astype('float32')
    scaled = scaler.fit_transform(values)

    # 获取训练集、测试集
    reframed = series_to_supervised(scaled, N_in, N_out)
    values = reframed.values
    n_train_data = int(k_train * len(values))  # 训练集的数量
    ind_train = np.random.randint(0, len(values), n_train_data)
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

    # case1:深度学习模型
    if Model_type == 'Dense':
        model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

    # case2:LSTM模型
    elif Model_type == 'LSTM':
        model.add(tf.keras.layers.LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))

    model.add(tf.keras.layers.Dense(N_out))  # 最后一层输出层
    model.compile(loss='mae', optimizer='adam')
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y),
                        verbose=2,
                        shuffle=False)
    plt.figure(figsize=(7, 7))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title("History of training")
    plt.legend()
    plt.savefig('./Output/history.svg')
    model.save('./Output/my_model.h5')  # 保存模型到文件
    model.summary()  # 打印网络结构


# endregion
disign_fit()


# region 3.评估模型
def eval(train_X, train_y, test_X, test_y):
    model = tf.keras.models.load_model('./Output/my_model.h5')  # 从文件读取模型
    # 预测值yhat与inv_yhat
    yhat = model.predict(test_X)
    if Model_type == 'LSTM':
        inv_yhat = scaler.inverse_transform(yhat)
    elif Model_type == 'Dense':
        inv_yhat = scaler.inverse_transform(yhat.reshape(yhat.shape[0], yhat.shape[2]))

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
    plt.title("Plot of test \nand predict load\n(randomly selected)")
    plt.legend()
    plt.subplot(1, 2, 2)
    ind = np.random.randint(0, len(test_y), 200)
    plt.hist(inv_yhat[ind].flatten(), bins=40, label='yhat')
    plt.hist(inv_y[ind].flatten(), bins=40, label='y')
    plt.title("Histogram of test \nand predict load\n(randomly selected)")
    plt.legend()
    plt.savefig('./Output/y_vs_yhat.svg')


# endregion
eval(train_X, train_y, test_X, test_y)

plt.show()
end = time.perf_counter()  # 程序结束运行时间
print("\nRunning time %d:%d:%.1f" % ((end - start) // 3600, ((end - start) % 3600) // 60, ((end - start) % 3600) % 60))
print("=============== Run to end successfully! ===============")

# region

# endregion
