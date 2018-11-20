import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # python3.x中返回一个iterable，而不是list
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def randCent(dataSet, k):
    n = np.shape(dataSet)[1]   # the number of features
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        # np.random.rand()返回一个或一组服从“0~1”均匀分布的随机样本值
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    # 第一列记录簇索引值，第二列存储误差：当前点到簇质心的距离
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        print(centroids)

        # 更新质心位置
        for cent in range(k):
            # np.nonzero(array):返回数组array中非零元素的索引值数组
            # 如果array是一个二维数组，则返回两个数组：
            #     第一个数组是按行维度来描述索引值，第二个数组是按列维度来描述索引值（都在0列）
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment


def showPlt(dataMat, alg=kMeans, numClust=4):
    myCentroids, clustAssing = alg(dataMat, numClust)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', '*', '^', '8', 'p', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    ax0.set_title('kMeans Algorithm')
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = dataMat[np.nonzero(clustAssing[:, 0].A == i)[0]]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        # flatten():返回一个折叠成一维的数组，该函数只能适用于numpy对象，即array或者mat，普通的list列表是不行的。
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A, ptsInCurrCluster[:, 1].flatten().A, marker=markerStyle, s=40)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=200)
    plt.show()


if __name__ == '__main__':
    dataMat = np.mat(loadDataSet('testSet.txt'))
    showPlt(dataMat)

