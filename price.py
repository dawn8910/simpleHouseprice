# _*_ coding: utf-8 _*_
"""
Time:     2021/10/27 19:52
Author:   LucifMonX
Version:  V 3.9
File:     例题1.2.py
Describe:
案例：假设你现在打算卖房子，想知道房子能卖多少钱？
我们拥有房子面积和卧室数量以及房子价格之间的对应数据 ex1data2.txt
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print('---------------------1.读取文件------------------------')
path = 'ex1data2.txt'
data = pd.read_csv(path, names=['size', 'bedroom', 'price'])
print(data.head())

# 分析：size数值过大，bedroom数值过小。这种不匹配的样本要进行归一化处理
print('---------------------2.特征值归一化--------------------------------')


def normalize_feature(data):
    return (data - data.mean()) / data.std()


data = normalize_feature(data)
print(data.head)

# 查看size和bedroom分别 和price 的关系
data.plot.scatter('size', 'price', label='size')
plt.show()

data.plot.scatter('bedroom', 'price', label='bedroom')
plt.show()
print('------------------------------3.添加全为1的列-----------------------------------')
data.insert(0, 'ones', 1)
print(data.head)
print('----------------------------------4.构造数据集----------------------------------------')
X = data.iloc[:, 0:-1]
print(X.head)
y = data.iloc[:, -1]
print(y.head)
print('-------------------------------5.将dataframe转成数组------------------------------------------')
X = X.values
print(X.shape)  # (47, 3)
y = y.values
print(y.shape)  # (47,)
y = y.reshape(47, 1)
print(y.shape)  # (47, 1)
print('------------------------------6.构造损失函数------------------------------------')


# X:特征值 y:标签 参数：theta
def computeCost(X, y, theta):
    inner = np.power((X @ theta - y), 2)
    return np.sum(inner) / (2 * len(X))


theta = np.zeros((3, 1))
cost_init = computeCost(X, y, theta)
print(cost_init)
print('-----------------------------7.梯度下降---------------------------------------')


def gradientDescent(X, y, theta, alpha, iters):
    costs = []
    for i in range(iters):
        # X.T:X的转置
        theta = theta - (X.T @ (X @ theta - y)) * alpha / len(X)
        cost = computeCost(X, y, theta)
        costs.append(cost)
    return theta, costs


# 不同alpha下的效果
candidate_alpha = [0.0003, 0.003, 0.03, 0.0001, 0.001, 0.01]
iters=2000
fig, ax = plt.subplots()
for alpha in candidate_alpha:
    _,costs=gradientDescent(X, y, theta, alpha, iters)
    ax.plot(np.arange((iters)),costs,label=alpha)
    ax.legend()
ax.set_xlabel('iters')
ax.set_ylabel('cost')
ax.set_title('cost vs. iters')
plt.show()
