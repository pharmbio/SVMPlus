import csv
import os
from sklearn.model_selection import train_test_split
from conformalClassification import icp
from conformalClassification import perf_measure as pm
from fileUtils import tuneWithCV as tune
from prettytable import PrettyTable
import numpy as np


svmFileName = "mnist8by8.csv"

svmPlusFileName = "mnistData.csv"

# Tuned values: 10, .01 : .915, 1, .01:914, 100,.01:915, 1000,.01,915
#
# tuned C and gamma kernel parameters
tunedSVMParam = [100, .000001]  # [10, .01]
tunedSVMStarParam = [100, .000001] #[1000,.01]
#tunedSVMPlus = [100, .001]
tunedSVMPlus = [10, .001]

nSample = 4000

def loadMNISTData(fileName, split = True):
    ifile = open(fileName)
    reader = csv.reader(ifile)
    a = []
    for row in reader:
        newRow = [float(val) for val in row]
        a.append(newRow)
    ifile.close()
    X = np.array([x for x in a]).astype(float)

    ifile = open("data/mnistLabel.csv")
    reader = csv.reader(ifile)
    a = []
    for row in reader:
        a.append(row)
    ifile.close()
    y = np.array(a).astype(float).reshape(1, nSample)
    y = np.array([x for x in y]).astype(int)
    y = y[0]

    if split:
        X_train, X_test, y_train, y_test, indices_train, indices_test = \
            train_test_split(X, y, range(len(X)), test_size=0.2,
                         stratify=y, random_state=7)
        return X_train, X_test, y_train, y_test, indices_train, indices_test

    return X, y


def prepareDataset(svmFile, svmPlusFile=None, returnAll = False):
    path = str("data/" + svmFile)
    X, X_test, y, y_test, indices_x, indices_test = \
        loadMNISTData(path, split=True)

    X_train, X_calib, y_train, y_calib, indices_train, indices_valid = \
        train_test_split(X, y, range(len(X)), test_size=0.3)

    if svmPlusFile is not None:
        XStar, yStar = loadMNISTData(str("data/" + svmPlusFile), split=False)
        XStar_test = XStar[indices_test]
        XStar = XStar[indices_x]
        XStar_train = XStar[indices_train]
        XStar_valid = XStar[indices_valid]
        yStar_test = yStar[indices_test]
        yStar = yStar[indices_x]
        yStar_train = yStar[indices_train]
        yStar_valid = yStar[indices_valid]

        if returnAll:
            return X_train, X_test, X_calib, y_train, y_test, y_calib, XStar_train,\
                   XStar_test, XStar_valid, yStar_train, yStar_test, yStar_valid
        return X_train, X_test, X_calib, y_train, y_test, y_calib, XStar_train

    return X_train, X_test, X_calib, y_train, y_test, y_calib



# Parameter estimation using grid search with validation set
def gridSearchWithCV(fileName):
    X_train, X_test, X_calib, y_train, y_test, y_calib = prepareDataset(fileName)
    tune.gridSearchWithCV(X_train, X_test, y_train, y_test, fileName)


# run SVM+ for sign descriptor files
def gridSearchSVMPlus(svmFile, svmPlusFile, kernelParam, kernelParamStar):
    X_train, X_test, X_calib, y_train, y_test, y_calib, XStar_train = prepareDataset(svmFile, svmPlusFile)
    tune.gridSearchSVMPlus(X_train, X_test, y_train, y_test,
                           XStar_train, svmFile, kernelParam=kernelParam, kernelParamStar=kernelParamStar)



# run SVM for finger print descriptor file
def ICPWithSVM(svmFile, C=10, gamma=.01):
    X_train, X_test, X_calib, y_train, y_test, y_calib = prepareDataset(svmFile)
    # record the results in a file
    dirPath = "icpWithSVMResults/"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    ofile = open(dirPath + svmFile, "a")
    ofile.write("Size of the train set: " + str(X_train.shape[0]) + "\n")
    ofile.write("Size of the test set: " + str(X_test.shape[0]) + "\n")
    # ICP
    pValues, y_predict = icp.ICPClassification(X_train, y_train, X_test,
                                    X_calib=X_calib, y_calib=y_calib)
    correct = np.sum(y_predict == y_test)
    accuracy = correct / len(y_predict)
    y_test = y_test.astype(int)
    y_test[y_test == -1] = 0

    errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(pValues, y_test)
    ofile.write("C = %f, gamma = %f, accuracy = %f \n" %
                (C, gamma, accuracy))
    ofile.write("errRate = %f, eff = %f, val = %f, obsFuzz = %f \n" %
                (errRate, eff, val, obsFuzz))
    ofile.close()
    return val, obsFuzz
    # pm.CalibrationPlot(pValues, y_test)
    # To be completed


# run SVM Plus for finger print descriptor file
def ICPWithSVMPlus(svmFile, svmPlusFile, C=10, gamma=.01,
                   kernelParam=0.01, kernelParamStar=0.01):
    X_train, X_test, X_calib, y_train, y_test, y_calib, XStar_train = \
        prepareDataset(svmFile, svmPlusFile)

    dirPath = "icpWithSVMPlusResults/"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    ofile = open(dirPath + svmFile, "a")
    ofile.write("Size of the train set: " + str(X_train.shape[0]) + "\n")
    ofile.write("Size of the test set: " + str(X_test.shape[0]) + "\n")

    pValues, y_predict  = icp.ICPClassification(X_train, y_train, X_test,
                                    XStar=XStar_train, C=C, gamma=gamma,
                                    K=kernelParam, KStar=kernelParamStar,
                                    X_calib=X_calib, y_calib=y_calib)
    correct = np.sum(y_predict == y_test)
    accuracy = correct / len(y_predict)

    y_test = y_test.astype(int)
    y_test[y_test == -1] = 0

    errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(pValues, y_test)
    # open again
    ofile.write("C = %f, gamma = %f, accuracy = %f \n" %
                (C, gamma, accuracy))
    ofile.write("errRate = %f, eff = %f, val = %f, obsFuzz = %f \n" %
                (errRate, eff, val, obsFuzz))

    ofile.close()
    return val, obsFuzz


# run SVM Plus for finger print descriptor file
def ICPWithSVMAndSVMPlus(svmFile, svmPlusFile):
    dirPath = "icpResults/"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    avgSvmVal = 0
    avgSvmOF = 0
    avgSvmStarVal = 0
    avgSvmStarOF = 0
    avgSvmPlusVal = 0
    avgSvmPlusOF = 0
    avgSvmAcc = 0
    avgSvmStarAcc = 0
    avgSvmPlusAcc = 0
    runs = 10
    for i in range(runs):
        X_train, X_test, X_calib, y_train, y_test, y_calib, \
        XStar_train, XStar_test, XStar_calib, yStar_train, yStar_test, yStar_calib = \
            prepareDataset(svmFile, svmPlusFile, returnAll=True)

        ofile = open(dirPath + svmFile, "a")
        ofile.write("Size of the train set: " + str(X_train.shape[0]) + "\n")
        ofile.write("Size of the test set: " + str(X_test.shape[0]) + "\n")
        y_test = y_test.astype(int)
        #SVM on X
        pValues, y_predict = icp.ICPClassification(X_train, y_train, X_test,
                                                   C= tunedSVMParam[0], gamma=tunedSVMParam[1],
                                                   X_calib=X_calib, y_calib=y_calib)
        correct = np.sum(y_predict == y_test)
        accuracy = correct / len(y_predict)

        y_test[y_test == -1] = 0

        errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(pValues, y_test)
        avgSvmVal = avgSvmVal+val
        avgSvmOF = avgSvmOF+obsFuzz
        avgSvmAcc = avgSvmAcc + accuracy
        ofile.write("C = %f, gamma = %f, accuracy = %f \n" %
                    (tunedSVMParam[0], tunedSVMParam[1], accuracy))
        ofile.write("errRate = %f, eff = %f, val = %f, obsFuzz = %f \n" %
                    (errRate, eff, val, obsFuzz))
        ofile.close()
        ofile = open(dirPath + svmFile, "a")
        y_test[y_test == 0] = -1

        # SVM on X Star
        pValues, y_predict = icp.ICPClassification(XStar_train, yStar_train, XStar_test,
                                                   C=tunedSVMStarParam[0], gamma=tunedSVMStarParam[1],
                                                   X_calib=XStar_calib, y_calib=yStar_calib)
        correct = np.sum(y_predict == y_test)
        accuracy = correct / len(y_predict)

        y_test[y_test == -1] = 0
        errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(pValues, y_test)
        avgSvmStarVal = avgSvmStarVal + val
        avgSvmStarOF = avgSvmStarOF + obsFuzz
        avgSvmStarAcc = avgSvmStarAcc + accuracy
        ofile.write("C = %f, gamma = %f, accuracy = %f \n" %
                    (tunedSVMStarParam[0], tunedSVMStarParam[1], accuracy))
        ofile.write("errRate = %f, eff = %f, val = %f, obsFuzz = %f \n" %
                    (errRate, eff, val, obsFuzz))
        ofile.close()
        ofile = open(dirPath + svmFile, "a")
        y_test[y_test == 0] = -1

        pValues, y_predict  = icp.ICPClassification(X_train, y_train, X_test,
                                        XStar=XStar_train, C=tunedSVMPlus[0], gamma=tunedSVMPlus[1],
                                        K=tunedSVMParam[1], KStar=tunedSVMStarParam[1],
                                        X_calib=X_calib, y_calib=y_calib)
        correct = np.sum(y_predict == y_test)
        accuracy = correct / len(y_predict)

        y_test[y_test == -1] = 0

        errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(pValues, y_test)
        avgSvmPlusVal = avgSvmPlusVal + val
        avgSvmPlusOF = avgSvmPlusOF + obsFuzz
        avgSvmPlusAcc = avgSvmPlusAcc + accuracy
        ofile.write("C = %f, gamma = %f, accuracy = %f \n" %
                    (tunedSVMPlus[0], tunedSVMPlus[1], accuracy))
        ofile.write("errRate = %f, eff = %f, val = %f, obsFuzz = %f \n" %
                    (errRate, eff, val, obsFuzz))

        ofile.close()

    ofile = open(dirPath + svmFile, "a")
    ofile.write("Average of SVM on X accuracy = %f, svmXValidity = %f, svmXOF = %f \n" %
                (avgSvmAcc/runs, avgSvmVal/runs, avgSvmOF/runs))
    ofile.write("Average of SVM on XStar accuracy = %f, svmXStartVal = %f, svmXStartOF = %f \n" %
                (avgSvmStarAcc/runs, avgSvmStarVal / runs, avgSvmStarOF / runs))
    ofile.write("Average of SVM Plus accuracy = %f, svmPlusVal = %f, svmPlusOF = %f \n" %
                (avgSvmPlusAcc/runs,  avgSvmPlusVal / runs, avgSvmPlusOF / runs))
    ofile.close()

    # pm.CalibrationPlot(pValues, y_test)





if __name__ == "__main__":
    ICPWithSVMAndSVMPlus(svmFileName, svmPlusFileName)




