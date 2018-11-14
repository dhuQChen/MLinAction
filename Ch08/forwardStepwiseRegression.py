import numpy as np
import matplotlib.pyplot as plt
import regression


# regularize by columns
def regularize(xMat):
    inMat = xMat.copy()
    inMeans = np.mean(inMat, 0)   # calc mean then subtract it off
    inVar = np.var(inMat, 0)      # calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat


def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        # print(ws.T)
        lowstError = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.getA(), yTest.getA())
                if rssE < lowstError:
                    lowstError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


xArr, yArr = regression.loadDataSet('abalone.txt')
# eps = 0.01
# stageWise(xArr, yArr, 0.01, 200)
# eps = 0.005
res = stageWise(xArr, yArr, 0.005, 1000)
plt.plot(res)
plt.show()
