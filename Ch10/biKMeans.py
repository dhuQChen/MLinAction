import numpy as np
import matplotlib.pyplot as plt
import kMeans


def biKMeans(dataSet, k, distMeas=kMeans.distEclud):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeas(np.mat(centroid0), dataSet[j, :]) ** 2

    while len(centList) < k:
        lowestSSE = np.inf
        # 遍历簇列表 centList 中每一个簇
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
            # 对当前簇进行 k=2 聚类，保存计算出的质心与误差值
            centroidMat, splitClustAss = kMeans.kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = np.sum(splitClustAss[:, 1])
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print('sseSplit, and sseNotSplit:', sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is:', bestCentToSplit)
        print('the len of bestClustAss:', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return np.mat(centList), clusterAssment


if __name__ == '__main__':
    dataMat = np.mat(kMeans.loadDataSet('testSet2.txt'))
    # centList, myNewAssments = biKMeans(dataMat, 3)
    # dataMat = np.mat(kMeans.loadDataSet('testSet.txt'))
    kMeans.showPlt(dataMat, alg=biKMeans, numClust=4)
