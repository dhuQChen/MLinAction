import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(np.float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(np.float(curLine[-1]))
    return dataMat, labelMat


def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    # Diagonal Matrix
    weights = np.mat(np.eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("This Matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k = 1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


if __name__ == '__main__':
    xArr, yArr = loadDataSet("ex0.txt")
    # you can choose different k
    # k = 1.0, k = 0.01, k = 0.003
    yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    # Why do we need sort for xMat ???
    xMat = np.mat(xArr)
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().getA(), np.mat(yArr).T.flatten().getA(), s=2, c='red')
    plt.show()






