# coding:utf8
'''
Created on Feb 16, 2011
Update on 2017-05-18
k Means Clustering for Ch10 of Machine Learning in Action
@author: Peter Harrington/那伊抹微笑
《机器学习实战》更新地址：https://github.com/apachecn/MachineLearning
'''
from numpy import *
import os
import numpy as np
import matplotlib.pyplot as plt

# 从文本中构建矩阵，加载文本文件，然后处理
def loadDataSet(fileName):    # 通用函数，用来解析以 tab 键分隔的 floats（浮点数）
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)    # 映射所有的元素为 float（浮点数）类型
        dataMat.append(fltLine)
    return dataMat


# 计算两个向量的欧式距离（可根据场景选择）
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) # la.norm(vecA-vecB)


# 为给定数据集构建一个包含 k 个随机质心的集合。随机质心必须要在整个数据集的边界之内，这可以通过找到数据集每一维的最小和最大值来完成。然后生成 0~1.0 之间的随机数并通过取值范围和最小值，以便确保随机点在数据的边界之内。
def randCent(dataSet, k):
    n = shape(dataSet)[1] # 列的数量
    centroids = mat(zeros((k,n))) # 创建k个质心矩阵
    for j in range(n): # 创建随机簇质心，并且在每一维的边界内
        minJ = min(dataSet[:,j])    # 最小值
        rangeJ = float(max(dataSet[:,j]) - minJ)    # 范围 = 最大值 - 最小值
        # print "max(dataSet[:,j]):", max(dataSet[:,j])
        # print "rangeJ", rangeJ
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))    # 随机生成
        print "\n---------->", centroids[:,j]
    return centroids


# k-means 聚类算法
# 该算法会创建k个质心，然后将每个点分配到最近的质心，再重新计算质心。
# 这个过程重复数次，知道数据点的簇分配结果不再改变位置。
# 运行结果（多次运行结果可能会不一样，可以试试，原因为随机质心的影响，但总的结果是对的， 因为数据足够相似，也可能会陷入局部最小值）
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]    # 行数
    clusterAssment = mat(zeros((m, 2)))    # 创建一个与 dataSet 行数一样，但是有两列的矩阵，用来保存簇分配结果
    # 簇索引 距离
    centroids = createCent(dataSet, k)    # 创建质心，随机k个质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):    # 循环每一个数据点并分配到最近的质心中去
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])    # 计算数据点到质心的距离
                if distJI < minDist:    # 如果距离比 minDist（最小距离）还小，更新 minDist（最小距离）和最小质心的 index（索引）
                    minDist = distJI; minIndex = j
            if clusterAssment[i, 0] != minIndex:    # 簇分配结果改变
                clusterChanged = True    # 簇改变
                clusterAssment[i, :] = minIndex,minDist**2    # 更新簇分配结果为最小质心的 index（索引），minDist（最小距离）的平方
        print centroids
        # print 'nonzero(clusterAssment[:, 0]:', nonzero(clusterAssment[:, 0])
        # print 'nonzero(clusterAssment[:, 0].A:', nonzero(clusterAssment[:, 0].A)
        # print 'nonzero(clusterAssment[:, 0].A==cent:', nonzero(clusterAssment[:, 0].A)


        for cent in range(k): # 更新质心
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A==cent)[0]] # 获取该簇中的所有点
            centroids[cent,:] = mean(ptsInClust, axis=0) # 将质心修改为簇中所有点的平均值，mean 就是求平均值的
    return centroids, clusterAssment

# 二分 KMeans 聚类算法, 基于 kMeans 基础之上的优化，以避免陷入局部最小值
def biKMeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    print "dataSet length:", m
    clusterAssment = mat(zeros((m,2))) # 保存每个数据点的簇分配结果和平方误差
    centroid0 = mean(dataSet, axis=0).tolist()[0] # 质心初始化为所有数据点的均值
    centList =[centroid0] # 初始化只有 1 个质心的 list
    for j in range(m): # 计算所有数据点到初始质心的距离平方误差
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k): # 当质心数量小于 k 时
        lowestSSE = inf
        for i in range(len(centList)): # 对每一个质心
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:] # 获取当前簇 i 下的所有数据点
            # 返回簇中心  每个点的簇中心以及到此中心的距离
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas) # 将当前簇 i 进行二分 kMeans 处理
            sseSplit = sum(splitClustAss[:,1]) # 将二分 kMeans 结果中的平方和的距离进行求和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1]) # 将未参与二分 kMeans 分配结果中的平方和的距离进行求和
            print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                # 簇中心
                bestNewCents = centroidMat
                # 每个点的簇中心以及到此中心的距离
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 找出最好的簇分配结果
        # bestClustAss 簇索引为0 的改为 bestCentToSplit
        # bestClustAss 簇索引为1 的改为 len(centList)
        print "before bestClustAss", bestClustAss
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) # 调用二分 kMeans 的结果，默认簇是 0,1. 当然也可以改成其它的数字
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit # 更新为最佳质心
        print "after bestClustAss", bestClustAss
        print 'nonzero(bestClustAss[:,0].A == 1)[0]:', nonzero(bestClustAss[:,0].A == 1)[0]
        print 'nonzero(bestClustAss[:,0].A == 0)[0]:', nonzero(bestClustAss[:,0].A == 0)[0]
        # print 'bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0]:', bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0]
        # print 'bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0]:', bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0]
        print 'the bestCentToSplit is: ', bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        # 更新质心列表
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0] # 更新原质心 list 中的第 i 个质心为使用二分 kMeans 后 bestNewCents 的第一个质心
        print "bestCentToSplit:", bestCentToSplit
        print "centList[bestCentToSplit]:", centList[bestCentToSplit]
        centList.append(bestNewCents[1,:].tolist()[0]) # 添加 bestNewCents 的第二个质心
        # 将此次迭代产生的bestClustAss替换为分类前的clusterAssment
        #####
        # 原始集合： 1,5 2,16 2,20
        # 对2,16 2,20进行二分划分，得到2,16 3,30 (bestCentToSplit=2)
        # 然后合并： 1,5 2,16 3,30
        ####
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss # 重新分配最好簇下的数据（质心）以及SSE
    return mat(centList), clusterAssment

def testBasicFunc():
    # 加载测试数据集
    datMat = mat(loadDataSet(os.getcwd() + "\\codetwo\\" + 'input/10.KMeans/testSet.txt'))

    # 测试 randCent() 函数是否正常运行。
    # 首先，先看一下矩阵中的最大值与最小值
    print 'min(datMat[:, 0])=', min(datMat[:, 0])
    print 'min(datMat[:, 1])=', min(datMat[:, 1])
    print 'max(datMat[:, 1])=', max(datMat[:, 1])
    print 'max(datMat[:, 0])=', max(datMat[:, 0])

    # 然后看看 randCent() 函数能否生成 min 到 max 之间的值
    print 'randCent(datMat, 2)=', randCent(datMat, 2)

    # 最后测试一下距离计算方法
    print ' distEclud(datMat[0], datMat[1])=', distEclud(datMat[0], datMat[1])

def testKMeans():
    # 加载测试数据集
    datMat = mat(loadDataSet(os.getcwd() + "\\codetwo\\" + 'input/10.KMeans/testSet.txt'))

    # 该算法会创建k个质心，然后将每个点分配到最近的质心，再重新计算质心。
    # 这个过程重复数次，知道数据点的簇分配结果不再改变位置。
    # 运行结果（多次运行结果可能会不一样，可以试试，原因为随机质心的影响，但总的结果是对的， 因为数据足够相似）
    myCentroids, clustAssing = kMeans(datMat, 4)
    # myCentroids 质心
    # clustAssing 簇索引

    print 'centroids=', myCentroids
    l1 = [x.tolist()[0][0] for x in clustAssing]
    # print l1
    # print clustAssing[0].tolist()[0][0]
    plt.scatter(np.array(datMat)[:, 1], np.array(datMat)[:, 0], c=l1)
    plt.scatter(myCentroids[:, 1].tolist(), myCentroids[:, 0].tolist(), c="r")
    plt.show()

def testBiKMeans():
    # 加载测试数据集
    datMat = mat(loadDataSet(os.getcwd() + "\\codetwo\\" + 'input/10.KMeans/testSet2.txt'))

    centList, myNewAssments = biKMeans(datMat, 4)

    print 'centList=', centList

    l1 = [x.tolist()[0][0] for x in myNewAssments]
    # print l1
    plt.scatter(np.array(datMat)[:, 1], np.array(datMat)[:, 0], c=l1)
    plt.scatter(centList[:, 1].tolist(), centList[:, 0].tolist(), c="r")
    plt.show()

if __name__ == "__main__":

    # 测试基础的函数
    # testBasicFunc()

    # 测试 kMeans 函数
    # testKMeans()

    # 测试二分 biKMeans 函数
    testBiKMeans()
