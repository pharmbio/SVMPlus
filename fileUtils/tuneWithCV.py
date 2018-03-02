import numpy as np
import svmPlusOpt as svmPlus
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import os



# Parameter tuning with cross validation
def gridSearchWithCV(X_train, X_test, y_train, y_test, logFile):
    paramC = [1, 10, 100, 1000]
    # paramC = [1.0, 2.154434690031884, 4.641588833612778, 10.0, 21.544346900318832, 46.4158883361278, 100.0, 215.44346900318845,
    # 464.15888336127773, 1000.0]
    paramGamma = [1e-5, 1e-4, 1e-3, 1e-2, .1]
    # paramGamma = [1e-05, 4.641588833612782e-06, 2.1544346900318822e-06, 1e-06, 4.641588833612782e-07, 2.1544346900318822e-07, 1e-07,
    # 4.641588833612773e-08, 2.1544346900318866e-08, 1e-08]

    # record the results in a file
    dirPath = "gridSearchResults/"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    ofile = open(dirPath + logFile, "a")
    ofile.write("Size of the train set: " + str(X_train.shape[0]) + "\n")
    ofile.write("Size of the test set: " + str(X_test.shape[0]) + "\n")
    ofile.close()
    predAcc = []
    index = []
    rocAUC = []
    n_splits = 5
    cv = StratifiedKFold(n_splits=n_splits)
    folds = [[train_index, test_index] for train_index, test_index in cv.split(X_train, y_train)]

    # Cross validate
    for i in range(len(paramC)):
        for j in range(len(paramGamma)):
            accuracy = 0
            areaUnderCurve = 0
            for fold in folds:
                cv_X_train = X_train[fold[0]]
                cv_X_test = X_train[fold[1]]
                cv_y_train = y_train[fold[0]]
                cv_y_test = y_train[fold[1]]
                clf = svm.SVC(gamma=paramGamma[j], C=paramC[i], probability=True)
                y_prob = clf.fit(cv_X_train, cv_y_train).predict_proba(cv_X_test)
                fpr, tpr, thresholds = roc_curve(cv_y_test, y_prob[:, 1])
                y_predict = clf.predict(cv_X_test)
                correct = np.sum(y_predict == cv_y_test)
                accuracy = accuracy + (correct / len(y_predict))
                areaUnderCurve = areaUnderCurve + auc(fpr, tpr)

            print(i, j)
            accuracy = accuracy / n_splits # averaging
            areaUnderCurve = areaUnderCurve / n_splits
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
def gridSearchSVMPlus(X_train, X_test, y_train, y_test, XStar_train, logFile,
                      kernelParam=0.0001, kernelParamStar=0.01):
    paramC = [1, 10, 100, 1000]
    paramGamma = [1e-5, 1e-4, 1e-3, 1e-2, .1, 1]

    dirPath = "gridSearchResults/"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    ofile = open(dirPath + "SVMPlus_" + logFile, "a")
    ofile.write("Size of the train set: " + str(X_train.shape[0]) + "\n")
    ofile.write("Size of the validation set: " + str(X_valid.shape[0]) + "\n")
    ofile.write("Size of the test set: " + str(X_test.shape[0]) + "\n")
    ofile.close()  # store file size and close, then open again

    predAcc = []
    index = []
    n_splits = 5
    cv = StratifiedKFold(n_splits=n_splits)
    folds = [[train_index, test_index] for train_index, test_index in cv.split(X, y)]

    for i in range(len(paramC)):
        for j in range(len(paramGamma)):
            accuracy = 0
            for fold in folds:
                cv_X_train = X_train[fold[0]]
                cv_X_test = X_train[fold[1]]
                cv_y_train = y_train[fold[0]]
                cv_y_test = y_train[fold[1]]
                # compute prediction accuracy using SVM+ on svmFile, and svmPlusFile2 as a priv-info
                clf = svmPlus.svmPlusOpt(cv_X_train, cv_y_train, XStar=XStar_train[fold[0]],
                                         C=paramC[i], gamma=paramGamma[j],
                                         kernel="rbf", kernelParam=kernelParam,
                                         kernelStar="rbf", kernelStarParam=kernelParamStar)
                y_predict = svmPlus.predict(cv_X_test, clf)
                correct = np.sum(y_predict == cv_y_test)
                accuracy = accuracy + (correct / len(y_predict))

            print(i, j)
            accuracy = accuracy / n_splits
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