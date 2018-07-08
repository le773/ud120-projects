# coding: utf-8

'''
Created on Jun 1, 2011
Update  on 2017-05-18
Author: Peter Harrington/片刻
GitHub：https://github.com/apachecn/MachineLearning
'''
from numpy import *
import matplotlib.pyplot as plt
import os
print(__doc__)


def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]
    return mat(datArr)


def pca(dataMat, topNfeat=9999999):
    """pca

    Args:
        dataMat   原数据集矩阵
        topNfeat  应用的N个特征
    Returns:
        lowDDataMat  降维后数据集
        reconMat     新的数据集空间
    """

    # 步骤1. 计算每一列的均值
    meanVals = mean(dataMat, axis=0)
    # dataMat: (1567L, 590L)
    print 'dataMat:', shape(dataMat)
    # print 'meanVals', meanVals

    # 每个向量同时都减去 均值
    meanRemoved = dataMat - meanVals
    # print 'meanRemoved=', meanRemoved

    # cov协方差=[(x1-x均值)*(y1-y均值)+(x2-x均值)*(y2-y均值)+...+(xn-x均值)*(yn-y均值)+]/(n-1)
    '''
    方差：（一维）度量两个随机变量关系的统计量
    协方差： （二维）度量各个维度偏离其均值的程度
    协方差矩阵：（多维）度量各个维度偏离其均值的程度

    当 cov(X, Y)>0时，表明X与Y正相关；(X越大，Y也越大；X越小Y，也越小。这种情况，我们称为“正相关”。)
    当 cov(X, Y)<0时，表明X与Y负相关；
    当 cov(X, Y)=0时，表明X与Y不相关。
    '''
    # 步骤2.
    covMat = cov(meanRemoved, rowvar=0)
    print 'covMat:', shape(covMat)
    # eigVals为特征值， eigVects为特征向量
    eigVals, eigVects = linalg.eig(mat(covMat))
    # print 'eigVals=', eigVals
    # print 'eigVects=', eigVects
    # 对特征值，进行从小到大的排序，返回从小到大的index序号
    # 特征值的逆序就可以得到topNfeat个最大的特征向量

    # # 步骤3.特征值排序索引，从小到大
    eigValInd = argsort(eigVals)
    # print 'eigValInd1=', eigValInd

    # -1表示倒序，返回topN的特征值[-1 到 -(topNfeat+1) 但是不包括-(topNfeat+1)本身的倒叙]
    # 特征索引
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    print 'eigValInd2=', eigValInd
    # 重组 eigVects 最大到最小
    print 'eigVects:', shape(eigVects)x
