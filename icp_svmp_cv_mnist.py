import numpy as np
import os
from sklearn.model_selection import train_test_split
from conformalClassification import icp
from conformalClassification import perf_measure as pm
from fileUtils import tuneWithCV as tune
from prettytable import PrettyTable
import csv


resizedFile = "data/mnist10by10.csv"

mnistFile = "data/mnistData.csv"

# tuned SVM C/kernel parameters
tunedSVMX = [100, .00001]  # [100, .1]
tunedSVMXStar = [100, .000001]  # 1000,.01
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



def prepareDataset(svmFile, svmPlusFile=None):
    X, X_test, y, y_test, indices_x, indices_test = \
        loadMNISTData(svmFile, split=True)

    X_train, X_valid, y_train, y_valid, indices_train, indices_valid = \
        train_test_split(X, y, range(len(X)), test_size=0.3, stratify=y, random_state=7)

    if svmPlusFile is not None:
        XStar, yStar = loadMNISTData(svmPlusFile, split=False)
        XStar = XStar[indices_x]
        XStar_train = XStar[indices_train]
        return X_train, X_test, X_valid, y_train, y_test, y_valid, XStar_train

    return X_train, X_test, X_valid, y_train, y_test, y_valid



# Parameter estimation using grid search with validation set
def gridSearchWithCV(fileName):
    X_train, X_test, X_valid, y_train, y_test, y_valid = prepareDataset(fileName)
    tune.gridSearchWithCV(X_train, X_test, y_train, y_test, fileName[-14:])


# run SVM+ for sign descriptor files
def gridSearchSVMPlus(svmFile, svmPlusFile,
                      kernelParam=0.0001, kernelParamStar=0.01):
    X_train, X_test, X_valid, y_train, y_test, y_valid, XStar_train = prepareDataset(svmFile, svmPlusFile)
    tune.gridSearchSVMPlus(X_train, X_test, y_train, y_test,
                           XStar_train, svmFile[-14:], kernelParam=kernelParam, kernelParamStar=kernelParamStar)


if __name__ == "__main__":
    #gridSearchWithCV(resizedFile)
    #gridSearchWithCV(mnistFile)

    gridSearchSVMPlus(resizedFile, mnistFile,
                      kernelParam=tunedSVMX[1],
                      kernelParamStar=tunedSVMXStar[1])

