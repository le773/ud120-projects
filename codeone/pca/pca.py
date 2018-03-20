# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from numpy import *
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# from .decision_regions import plot_decision_regions

df_wine = pd.read_csv(os.getcwd() + "\\codeone\\" + 'input/pca/wine.data', header=None) # 加载葡萄酒数据集


def plot_decision_regions(X, y, classifier,
                          test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot all samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                    alpha=1.0, linewidths=1, marker='o',edgecolor='black',
                    s=55, label='test set')


from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values # 把数据与标签拆分开来
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0) # 把整个数据集的70%分为训练集，30%为测试集

# 下面3行代码把数据集标准化为单位方差和0均值
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)
print "X_train_std:\n", shape(X_train_std)


cov_mat = np.cov(X_train_std.T)
print cov_mat.shape # 输出为(13, 13）

eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print "eigen_vals:\n", eigen_vals
print "eigen_vecs:\n", shape(eigen_vecs)
print "eigen_vecs shape:\n", eigen_vecs.shape

# 求出特征值的和
tot = sum(eigen_vals)

# 求出每个特征值占的比例（降序）
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]

# 返回var_exp的累积和
cum_var_exp = np.cumsum(var_exp)

print 'cum_var_exp:\n',cum_var_exp

# 下面的代码都是绘图的，涉及的参数建议去查看官方文档
if 0:
    plt.bar(range(len(eigen_vals)), var_exp, width=1.0, bottom=0.0, alpha=0.5, label='individual explained variance')
    plt.step(range(len(eigen_vals)), cum_var_exp, where='post', label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.show()


eigen_pairs =[(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))] # 把特征值和对应的特征向量组成对
eigen_pairs.sort(reverse=True) # 用特征值排序

print "eigen_pairs:\n", eigen_pairs

first = eigen_pairs[0][1]
second = eigen_pairs[1][1]

print 'first>', first
# 一维长度为len的数组转为len,1的二维数组
first = first[:,np.newaxis]
print 'first>', first
second = second[:,np.newaxis]

# first second 拼接为len,2的二维数组
W = np.hstack((first,second))
print 'W>',W

# X_train *  eigen_vecs = y_train
X_train_pca = X_train_std.dot(W) # 转换训练集

print 'y_train>', y_train
print "X_train_std>",X_train_std.shape
print "W>",W.shape
print "X_train_pca>",X_train_pca.shape
print "y_train>",y_train.shape
if 0:
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_pca[y_train==l, 0], X_train_pca[y_train==l, 1], c=c, label=l, marker=m) # 散点图
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
pca = PCA(n_components=2) # 保留2个主成分
lr = LogisticRegression() # 创建逻辑回归对象
X_train_pca = pca.fit_transform(X_train_std) # 把原始训练集映射到主成分组成的子空间中
# 可释方差
# 每一个主成分占数据变动的比例
print 'explained_variance_ratio_:', pca.explained_variance_ratio_
X_test_pca = pca.transform(X_test_std) # 把原始测试集映射到主成分组成的子空间中

# 从pca的成分属性中获取主成分的数据，也就是重要程度
first_pc = pca.components_[0] # 也即 first
second_pc = pca.components_[1]

print 'first_pc:', first_pc
print 'second_pc:', second_pc


lr.fit(X_train_pca, y_train) # 用逻辑回归拟合数据
plot_decision_regions(X_train_pca, y_train, classifier=lr)
lr.score(X_test_pca, y_test) # 0.98 在测试集上的平均正确率为0.98
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()
