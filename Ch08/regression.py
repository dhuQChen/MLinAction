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


def standRegress(xArr, yArr):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse\n")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


if __name__ == '__main__':
    # DataSet
    xArr, yArr = loadDataSet('ex0.txt')
    ws = standRegress(xArr, yArr)

    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    # prediction
    yHat = xMat * ws

    # Data Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().getA(), yMat.T[:, 0].flatten().getA(), s=15, c='b')
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat, c='r')
    plt.show()

    # Correlation coefficient
    print(np.corrcoef(yHat.T, yMat))





