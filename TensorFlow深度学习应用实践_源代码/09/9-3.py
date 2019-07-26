"""
详细建立了一个BP反馈神经网络模型，共有三层（输入+隐藏+输出），
训练集的输入为cases = [[0, 0], [0, 1], [1, 0], [1, 1]]，
训练集输出为labels = [[0], [1], [1], [0]]，
训练神经网络limit=100次，并计算输入为cases时的输出
"""

import numpy as np
import math
import random

# region 功能函数
# 生成[a, b]内的随机数
def rand(a, b):
    return (b - a) * random.random() + a


# 生成mxn的0矩阵
def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


# 激活函数，f(x)
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# 激活函数的导数，f'(x)
def sigmod_derivate(x):
    return sigmoid(x) * (1 - sigmoid(x))


# endregion

class BPNeuralNetwork:
    # (1)init初始化
    def __init__(self):
        self.input_n = 0  # 输入层节点数
        self.hidden_n = 0
        self.output_n = 0

        self.input_cells = []  # 各层节点数值
        self.hidden_cells = []
        self.output_cells = []

        self.input_weights = []
        self.output_weights = []

    # (2)对init中的数据初始化
    def setup(self, ni, nh, no):
        self.input_n = ni + 1  # 输入层节点数，包含bias
        self.hidden_n = nh
        self.output_n = no

        self.input_cells = [1.0] * self.input_n  # 初始化各层节点数值，为[1, 1, 1, ... , 1]，长度为input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n

        self.input_weights = make_matrix(self.input_n, self.hidden_n)  # 初始化权重矩阵结构，尚未赋值
        self.output_weights = make_matrix(self.hidden_n, self.output_n)

        # random activate 为权重矩阵赋值
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)

    # (3)神经网络前向计算，返回输出层的结果
    def predict(self, inputs):

        # 计算输入层节点值
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]

        # 计算隐藏层
        # https://github.com/GetMyPower/mypython/blob/master/TensorFlow深度学习应用实践_源代码/09/BP算法.md
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)

        # 计算输出层
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)

        return self.output_cells[:]

    # (4)反馈传播，inputs=case，训练数据因变量为label，
    def back_propagate(self, case, label, learn):

        # 前向计算各神经元输出
        self.predict(case)
        # 计算输出层的误差
        output_deltas = [0.0] * self.output_n
        for k in range(self.output_n):
            error = label[k] - self.output_cells[k]
            output_deltas[k] = sigmod_derivate(self.output_cells[k]) * error

        # 计算隐藏层的误差
        hidden_deltas = [0.0] * self.hidden_n
        for j in range(self.hidden_n):
            error = 0.0
            for k in range(self.output_n):
                error += output_deltas[k] * self.output_weights[j][k]
            hidden_deltas[j] = sigmod_derivate(self.hidden_cells[j]) * error  # 相当于δ*f'(e)

        # 更新输出层权重，learn就是η，相当于w' = w + η*δ*f'(e)*y
        for j in range(self.hidden_n):
            for k in range(self.output_n):
                self.output_weights[j][k] += learn * output_deltas[k] * self.hidden_cells[j]

        # 更新隐藏层权重
        for i in range(self.input_n):
            for j in range(self.hidden_n):
                self.input_weights[i][j] += learn * hidden_deltas[j] * self.input_cells[i]

        # 可能是 0.5 * Σ( f(θ)-yi )^2
        error = 0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2

        return error

    # (5)将神经网络训练limit次
    def train(self, cases, labels, limit=100, learn=0.05):
        for i in range(limit):
            error = 0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn)
        pass

    # (6)启动网络训练，并输出计算结果
    def test(self):
        cases = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
        labels = [[0], [1], [1], [0]]
        self.setup(2, 5, 1)
        self.train(cases, labels, 10000, 0.05)
        for case in cases:
            print(self.predict(case))


if __name__ == '__main__':
    nn = BPNeuralNetwork()
    nn.test()

