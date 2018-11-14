import numpy as np
import regression


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


def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


abX, abY = loadDataSet('abalone.txt')
yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
print('\n----Effect on Training Data-----')
print('k=0.1:', rssError(abY[0:99], yHat01.T), '  k=1:', rssError(abY[0:99], yHat1.T),
      '  k=10:', rssError(abY[0:99], yHat10.T))


yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
print('\n----Effect on Validation Data-----')
print('k=0.1: ', rssError(abY[100:199], yHat01.T), '  k=1: ', rssError(abY[100:199], yHat1.T),
      '  k=10: ', rssError(abY[100:199], yHat10.T))


ws = regression.standRegress(abX[0:99], abY[0:99])
yHat = np.mat(abX[100:199]) * ws
print('\n----Simple Linear Regression------')
print(rssError(abY[100:199], yHat.T.getA()))




