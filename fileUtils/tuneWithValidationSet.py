import numpy as np
import svmPlusOpt as svmPlus
from sklearn import svm
from fileUtils import csrFormat as csr
from fileUtils import csvFormat as csv
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import os
from sklearn.model_selection import train_test_split
from conformalClassification import icp
from conformalClassification import perf_measure as pm
from prettytable import PrettyTable



# Parameter estimation using grid search with validation set
def gridSearchWithValidation(X_train, X_test, X_valid, y_train, y_test, y_valid, logFile):
    paramC = [1, 10, 100]
    paramGamma = [1e-4, 1e-3, 1e-2, .1]

    # record the results in a file
    dirPath = "gridValidationResults/"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    ofile = open(dirPath + logFile, "a")
    ofile.write("Size of the train set: " + str(X_train.shape[0]) + "\n")
    ofile.write("Size of the validation set: " + str(X_valid.shape[0]) + "\n")
    ofile.write("Size of the test set: " + str(X_test.shape[0]) + "\n")
    ofile.close()
    predAcc = []
    index = []
    rocAUC = []
    # Cross validate
    for i in range(len(paramC)):
        for j in range(len(paramGamma)):
            clf = svm.SVC(gamma=paramGamma[j], C=paramC[i], probability=True)
            y_prob = clf.fit(X_train, y_train).predict_proba(X_valid)
            fpr, tpr, thresholds = roc_curve(y_valid, y_prob[:, 1])
            y_predict = clf.predict(X_valid)
            correct = np.sum(y_predict == y_valid)
            accuracy = correct / len(y_predict)
            areaUnderCurve = auc(fpr, tpr)
            print(i, j)
            rocAUC.append(areaUnderCurve)
            predAcc.append(accuracy)
            index.append([i, j])
            # open again
            ofile = open(dirPath + logFile, "a")
            ofile.write("param C = %f, gamma = %f, mean pred accuracy = %f, mean AUC = %f \n" %
                        (paramC[i], paramGamma[j], accuracy, areaUnderCurve))
            ofile.close()
    selectedIndex = np.argmax(predAcc)
    C = paramC[index[selectedIndex][0]]
    gamma = paramGamma[index[selectedIndex][1]]

    # Fit the SVM on whole training set and calculate results on test set
    clf = svm.SVC(gamma=gamma, C=C)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)
    predAcc = round(correct / len(y_predict), 3)
    print("param C = %f, gamma = %f, pred accuracy = %f \n" % (C, gamma, predAcc))
    # open again
    ofile = open(dirPath + logFile, "a")
    ofile.write("param C = %f, gamma = %f, pred accuracy = %f \n" % (C, gamma, predAcc))

    # select model based on AUC
    selectedIndex = np.argmax(rocAUC)
    C = paramC[index[selectedIndex][0]]
    gamma = paramGamma[index[selectedIndex][1]]

    # Fit the SVM on whole training set and calculate AUC for test set
    clf = svm.SVC(gamma=gamma, C=C, probability=True)
    y_prob = clf.fit(X_train, y_train).predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    finalAUC = auc(fpr, tpr)
    # print("param C = %f, gamma = %f, AUC = %f \n" % (C, gamma, finalAUC))
    ofile.write("param C = %f, gamma = %f, AUC = %f \n" % (C, gamma, finalAUC))
    ofile.close()
    
    
    
# run SVM+ for sign descriptor files
def gridSearchSVMPlus(X_train, X_test, X_valid, y_train, y_test, y_valid, XStar_train, logFile,
                      kernelParam=0.0001, kernelParamStar=0.01):
    paramC = [.1, 1, 10]
    paramGamma = [1e-3, 1e-2, .1]
    
    dirPath = "gridValidationResults/"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    ofile = open(dirPath + "SVMPlus_" + logFile, "a")
    ofile.write("Size of the train set: " + str(X_train.shape[0]) + "\n")
    ofile.write("Size of the validation set: " + str(X_valid.shape[0]) + "\n")
    ofile.write("Size of the test set: " + str(X_test.shape[0]) + "\n")
    ofile.close()  # store file size and close, then open again

    predAcc = []
    index = []
    for i in range(len(paramC)):
        for j in range(len(paramGamma)):
            accuracy = 0
            # compute prediction accuracy using SVM+ on logFile, and svmPlusFile2 as a priv-info
            clf = svmPlus.svmPlusOpt(X_train, y_train, XStar=XStar_train,
                                     C=paramC[i], gamma=paramGamma[j],
                                     kernel="rbf", kernelParam=kernelParam,
                                     kernelStar="rbf", kernelStarParam=kernelParamStar)
            y_predict = svmPlus.predict(X_valid, clf)
            correct = np.sum(y_predict == y_valid)
            accuracy = correct / len(y_predict)
            print(i, j)
            predAcc.append(accuracy)
            index.append([i, j])
            ofile = open(dirPath + "SVMPlus_" + logFile, "a")
            ofile.write("param C = %f, gamma = %f, mean pred accuracy = %f \n" %
                        (paramC[i], paramGamma[j], accuracy))
            ofile.close()

    selectedIndex = np.argmax(predAcc)
    C = paramC[index[selectedIndex][0]]
    gamma = paramGamma[index[selectedIndex][1]]
    # For optimal parameters
    clf = svmPlus.svmPlusOpt(X_train, y_train, XStar=XStar_train, C=C, gamma=gamma,
                             kernel="rbf", kernelParam=kernelParam,
                             kernelStar="rbf", kernelStarParam=kernelParamStar)
    y_predict = svmPlus.predict(X_test, clf)
    correct = np.sum(y_predict == y_test)
    predAcc = round(correct / len(y_predict), 3)
    ofile = open(dirPath + "SVMPlus_" + logFile, "a")
    ofile.write("param C = %f, gamma = %f, pred Acc = %f \n" % (C, gamma, predAcc))
    ofile.close()