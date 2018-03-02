import numpy as np
import os
from sklearn.model_selection import train_test_split
from conformalClassification import icp
from conformalClassification import perf_measure as pm
from fileUtils import tuneWithValidationSet as tune
from prettytable import PrettyTable
import csv


resizedFile = "data/mnistResizedData.csv"

mnistFile = "data/mnistData.csv"

# tuned SVM C/kernel parameters
tunedSVMX = [10, .00001]  # [100, .1]
tunedSVMXStar = [10, .000001]  # 1000,.01
tunedSVMPlus = [10, .1]

def loadMNISTData(fileName, split = False):
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
    y = np.array(a).astype(float).reshape(1, 4000)
    y = np.array([x for x in y]).astype(int)
    y = y[0]

    if split:
        X_train, X_test, y_train, y_test, indices_train, indices_test = \
            train_test_split(X, y, range(len(X)), test_size=0.2,
                         stratify=y, random_state=7)
        return X_train, X_test, y_train, y_test, indices_train, indices_test

    return X, y

def prepareDataset(svmFile, svmPlusFile=None):
    X, X_test, y, y_test, indices_x, indices_test = \
        loadMNISTData(svmFile, split=True)

    X_train, X_valid, y_train, y_valid, indices_train, indices_valid = \
        train_test_split(X, y, range(len(X)), test_size=0.75,
                         stratify=y, random_state=7)

    if svmPlusFile is not None:
        XStar, yStar = loadMNISTData(svmPlusFile, split=False)
        XStar = XStar[indices_x]
        XStar_train = XStar[indices_train]
        return X_train, X_test, X_valid, y_train, y_test, y_valid, XStar_train

    return X_train, X_test, X_valid, y_train, y_test, y_valid


# Parameter estimation using grid search with validation set
def gridSearchWithValidation(fileName):
    X_train, X_test, X_valid, y_train, y_test, y_valid = prepareDataset(fileName)
    tune.gridSearchWithValidation(X_train, X_test, X_valid, y_train, y_test, y_valid, fileName[-14:])


# run SVM+ for sign descriptor files
def gridSearchSVMPlus(svmFile, svmPlusFile,
                      kernelParam=0.0001, kernelParamStar=0.01):
    X_train, X_test, X_valid, y_train, y_test, y_valid, XStar_train = prepareDataset(svmFile, svmPlusFile)
    tune.gridSearchSVMPlus(X_train, X_test, X_valid, y_train, y_test, y_valid,
                           XStar_train, svmFile[-15:], kernelParam=kernelParam, kernelParamStar=kernelParamStar)


# run SVM for finger print descriptor file
def ICPWithSVM(svmFile, C=10, gamma=.01):
    X_train, X_test, X_valid, y_train, y_test, y_valid = prepareDataset(svmFile)
    # record the results in a file
    dirPath = "icpWithSVMResults/"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    ofile = open(dirPath + "mnist", "a")
    ofile.write("Size of the train set: " + str(X_train.shape[0]) + "\n")
    ofile.write("Size of the test set: " + str(X_test.shape[0]) + "\n")
    # ICP
    pValues = icp.ICPClassification(X_train, y_train, X_test,
                                    X_calib=X_valid, y_calib=y_valid)
    y_test = y_test.astype(int)
    y_test[y_test == -1] = 0

    errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(pValues, y_test)
    # open again
    ofile.write("SVM on \n")
    ofile.write("C = %f, gamma = %f, errRate = %f, eff = %f, val = %f, obsFuzz = %f \n" %
                (C, gamma, errRate, eff, val, obsFuzz))
    ofile.close()
    return val, obsFuzz
    # pm.CalibrationPlot(pValues, y_test)
    # To be completed


# run SVM Plus for finger print descriptor file
def ICPWithSVMPlus(svmFile, svmPlusFile, C=10, gamma=.01,
                   kernelParam=0.01, kernelParamStar=0.01):
    X_train, X_test, X_valid, y_train, y_test, y_valid, XStar_train = \
        prepareDataset(svmFile, svmPlusFile)

    dirPath = "icpWithSVMPlusResults/"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    ofile = open(dirPath + "SvmPlust_mnist", "a")
    ofile.write("Size of the train set: " + str(X_train.shape[0]) + "\n")
    ofile.write("Size of the test set: " + str(X_test.shape[0]) + "\n")

    pValues = icp.ICPClassification(X_train, y_train, X_test,
                                    XStar=XStar_train, C=C, gamma=gamma,
                                    K=kernelParam, KStar=kernelParamStar,
                                    X_calib=X_valid, y_calib=y_valid)
    y_test = y_test.astype(int)
    y_test[y_test == -1] = 0

    errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(pValues, y_test)
    # open again
    ofile.write("C = %f, gamma = %f, errRate = %f, eff = %f, val = %f, obsFuzz = %f \n" %
                (C, gamma, errRate, eff, val, obsFuzz))

    ofile.close()
    return val, obsFuzz
    # pm.CalibrationPlot(pValues, y_test)

    # To be completed


# some settings before experiment

if __name__ == "__main__":
    #gridSearchWithValidation(resizedFile)
    #gridSearchWithValidation(mnistFile)
    # tune SVM/SVM+ parameters
    gridSearchSVMPlus(resizedFile, mnistFile,
                      kernelParam=tunedSVMX[1],
                      kernelParamStar=tunedSVMXStar[1])


    if 0:
        val, obsFuzz = ICPWithSVM(resizedFile,
                                  C=10, gamma=.00001)
        t = PrettyTable(['Dataset-BURSI', 'M/L Method', 'Validity', 'Obs-fuzziness'])
        t.add_row(['phys-chem', 'SVM', val, obsFuzz])
        print(t)

if 0:
    val, obsFuzz = ICPWithSVM(resizedFile,
                              C=tunedSVMX[0], gamma=tunedSVMX[1])
    t = PrettyTable(['Dataset-BURSI', 'M/L Method', 'Validity', 'Obs-fuzziness'])
    t.add_row(['phys-chem', 'SVM', val, obsFuzz])

    val, obsFuzz = ICPWithSVM(mnistFile, C=tunedSVMXStar[0], gamma=tunedSVMXStar[1])
    t.add_row(['morgan', 'SVM', val, obsFuzz])
    val, obsFuzz = ICPWithSVMPlus(resizedFile, mnistFile,
                                  C=tunedSVMPlus[0], gamma=tunedSVMPlus[1],
                                  kernelParam=tunedSVMX[1], kernelParamStar=tunedSVMXStar[1])
    t.add_row(['phys-chem+morgan', 'SVM+', val, obsFuzz])

    print(t)


