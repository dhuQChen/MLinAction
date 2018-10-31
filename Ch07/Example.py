import Adaboost
import numpy as np


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
classifierArray = Adaboost.adaBoostTrainDS(dataArr, labelArr, 50)
testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
prediction = Adaboost.adaClassify(testArr, classifierArray)
errArr = np.mat(np.ones((67, 1)))
print("misclassified rate", errArr[prediction != np.mat(testLabelArr).T].sum() / 67)

