import numpy as np


def loadSimpData():
    datMat = np.matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


# classify by threshVal
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':  # less
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:   # greater
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    '''
    return:
        bestStump: divide by dimension, thresVal, and inequal
        minError: the error of bestStump
        bestClassEst: the prediction of bestStump
    '''
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    # loop over all dimensions
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        # loop over all range in current dimension
        for j in range(-1, int(numSteps) + 1):

            # go over less than and greater than
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)

                # call stump classify with i, j, lessThan
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0

                # calc total error multiplied by D
                weightedError = D.T * errArr
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f"
                      % (i, threshVal, inequal, weightedError))

                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst


def adaBoostTrainDS(dataMat, classLabels, numIt=40):
    weekClassArr = []
    m = np.shape(dataMat)[0]
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataMat, classLabels, D)
        # classEst: m * 1, D: m * 1
        print("D: ", D.T)
        alpha = np.float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weekClassArr.append(bestStump)
        print("classEst:", classEst.T)

        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        print("aggClassEst: ", aggClassEst.T)

        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate, "\n")
        if errorRate == 0.0:
            break
    return weekClassArr


def adaClassify(dataToClass, classifierArr):
    dataMatrix = np.mat(dataToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    print(np.sign(aggClassEst))
    return np.sign(aggClassEst)


dataMat, classLabels = loadSimpData()
classifierArr = adaBoostTrainDS(dataMat, classLabels, 9)
# adaClassify([0, 0], classifierArr)
adaClassify([[5, 5], [0, 0]], classifierArr)
