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
from testCode import MNISTDataset as mnist
from prettytable import PrettyTable


phyChemFile = ["bursi_nosalts_molsign.sdf.txt_descriptors.csv",
               "sr-mmp_nosalt.sdf.std_nodupl_class.sdf_descriptors.csv",
               "smiles_cas_N6512.sdf.std_class.sdf_descriptors.csv"]

morganUnhashedFiles = ["bursi_nosalts_molsign.sdf.txt_SVMLIGHT_Morgan_unhashed_radius_3.csv",
                       "sr-mmp_nosalt.sdf.std_nodupl_class.sdf_SVMLIGHT_Morgan_unhashed_radius_3.csv",
                       "smiles_cas_N6512.sdf.std_class.sdf_SVMLIGHT_Morgan_unhashed_radius_3.csv"]

PHYSCHEM = 0
UNHASHED = 1

BURSI = 0
MMP = 1
CAS = 2

# tuned kernel parameters
tunedParam = [[.1, .01],  # BURSI
              [.1, .01],  # MMP
              [.1, .01]]  # CAS #[.01, .01]]  # CAS

# tuned kernel parameters
tunedCosts = [100, 10, 100]  # CAS


# Parameter estimation using grid search with validation set
def gridSearchWithValidation(fileName):
    paramC = [1, 10, 100]
    paramGamma = [1e-4, 1e-3, 1e-2, .1]

    # X, X_test, y, y_test =  loadDataset(fileName, split = True) # train(80):test(20) split
    path = str("/home/niharika/PycharmProjects/SVMPlus/MorganDataset/" + fileName)
    try:
        X, X_test, y, y_test = csr.readCSRFile(path, split=True)
    except:
        X, X_test, y, y_test = csv.loadDataset(path, split=True)

    X_train, X_valid, y_train, y_valid, = \
        train_test_split(X, y, test_size=0.2, stratify=y, random_state=7)

    # record the results in a file
    dirPath = "gridValidationResults/"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    ofile = open(dirPath + fileName, "a")
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
            ofile = open(dirPath + fileName, "a")
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
    ofile = open(dirPath + fileName, "a")
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
def gridSearchSVMPlus(svmFile, svmPlusFile,
                      kernelParam=0.0001, kernelParamStar=0.01):
    paramC = [.1, 1, 10]
    paramGamma = [1e-3, 1e-2, .1]
    path = str("/home/niharika/PycharmProjects/SVMPlus/MorganDataset/" + svmFile)
    try:
        X, X_test, y, y_test, indices_main, indices_test = \
            csr.readCSRFile(path, split=True, returnIndices=True)
    except:
        X, X_test, y, y_test, indices_main, indices_test = \
            csv.loadDataset(path, split=True, returnIndices=True)

    X_train, X_valid, y_train, y_valid, indices_train, indices_valid = \
        train_test_split(X, y, range(len(X)), test_size=0.2, stratify=y, random_state=7)

    path = str("MorganDataset/" + svmPlusFile)
    XStar, yStar = csr.readCSRFile(path)

    XStar_test = XStar[indices_test]
    XStar = XStar[indices_main]
    yStar_test = yStar[indices_test]
    yStar = yStar[indices_main]

    XStar_valid = XStar[indices_valid]
    XStar_train = XStar[indices_train]
    yStar_valid = yStar[indices_valid]
    yStar_train = yStar[indices_train]

    if (y == yStar).all() and (y_test == yStar_test).all():
        print("same split for both the files")
    else:
        sys.exit("different split for X and XStar")

    predAcc = []
    index = []
    dirPath = "gridValidationResults/"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    ofile = open(dirPath + "SVMPlus_" + svmFile, "a")
    ofile.write("Size of the train set: " + str(X_train.shape[0]) + "\n")
    ofile.write("Size of the validation set: " + str(X_valid.shape[0]) + "\n")
    ofile.write("Size of the test set: " + str(X_test.shape[0]) + "\n")
    ofile.close()  # store file size and close, then open again

    for i in range(len(paramC)):
        for j in range(len(paramGamma)):
            accuracy = 0
            # compute prediction accuracy using SVM+ on svmFile, and svmPlusFile2 as a priv-info
            clf = svmPlus.svmPlusOpt(X_train, y_train, XStar=XStar_train,
                                     C=paramC[i], gamma=paramGamma[j],
                                     kernel="rbf", kernelParam=kernelParam,
                                     kernelStar="rbf", kernelStarParam=kernelParamStar)
            y_predict = svmPlus.predict(X_test, clf)
            correct = np.sum(y_predict == y_test)
            accuracy = correct / len(y_predict)
            print(i, j)
            predAcc.append(accuracy)
            index.append([i, j])
            ofile = open(dirPath + "SVMPlus_" + svmFile, "a")
            ofile.write("param C = %f, gamma = %f, mean pred accuracy = %f \n" %
                        (paramC[i], paramGamma[j], accuracy))
            ofile.close()

    selectedIndex = np.argmax(predAcc)
    C = paramC[index[selectedIndex][0]]
    gamma = paramGamma[index[selectedIndex][1]]
    # For optimal parameters
    clf = svmPlus.svmPlusOpt(X, y, XStar=XStar, C=C, gamma=gamma,
                             kernel="rbf", kernelParam=kernelParam,
                             kernelStar="rbf", kernelStarParam=kernelParamStar)
    y_predict = svmPlus.predict(X_test, clf)
    correct = np.sum(y_predict == y_test)
    predAcc = round(correct / len(y_predict), 3)
    ofile = open(dirPath + "SVMPlus_" + svmFile, "a")
    ofile.write("param C = %f, gamma = %f, pred Acc = %f \n" % (C, gamma, predAcc))
    ofile.close()

'''
    # train and predict using SVM implemented in sklearn
    clf = svm.SVC(gamma=.000001, C=100.)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print("Prediction accuracy of SVM on X")
    print("number of test error = %d, test size = %d" % (len(y_predict) - correct, len(y_predict)))
'''
# run SVM for finger print descriptor file
def ICPWithSVM(svmFile  = "mnist", C=10, gamma=.01):
    X_train, X_test, y_train, y_test, XStar_train, XStar_test, X_valid, y_valid, XStar_valid = \
        mnist.loadMNISTData()
    # record the results in a file
    dirPath = "icpWithSVMResults/"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    ofile = open(dirPath + svmFile, "a")
    ofile.write("Size of the train set: " + str(X_train.shape[0]) + "\n")
    ofile.write("Size of the test set: " + str(X_test.shape[0]) + "\n")
    t = PrettyTable(['Dataset-MNIST', 'M/L Method', 'Validity', 'Obs-fuzziness'])
    # ICP
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    C1 = 100
    gamma1 = .000001
    pValues = icp.ICPClassification(X_train, y_train, X_test, C=C1, gamma=gamma1,
                                    X_calib= X_valid, y_calib= y_valid)
    y_test[y_test == -1] = 0

    errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(pValues, y_test)
    t.add_row(['low-reso', 'SVM', val, obsFuzz])
    ofile.write("SVM on X\n")
    ofile.write("C = %f, gamma = %f, errRate = %f, eff = %f, val = %f, obsFuzz = %f \n" %
                (C1, gamma1, errRate, eff, val, obsFuzz))

    C2 = 100
    gamma2 = 0.000001
    pValues = icp.ICPClassification(XStar_train, y_train, XStar_test,
                                    C = C2, gamma = gamma2, X_calib= XStar_valid, y_calib= y_valid)

    errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(pValues, y_test)
    t.add_row(['high-reso', 'SVM', val, obsFuzz])
    # open again
    ofile.write("SVM on XStar\n")
    ofile.write("C = %f, gamma = %f, errRate = %f, eff = %f, val = %f, obsFuzz = %f \n" %
                (C2, gamma2, errRate, eff, val, obsFuzz))

    C = 13
    gamma = .001
    pValues = icp.ICPClassification(X_train, y_train, X_test, XStar=XStar_train,
                                    C=C, gamma=gamma, K = 0.000001, KStar= 0.000001,
                                    X_calib=X_valid, y_calib=y_valid)

    errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(pValues, y_test)
    t.add_row(['low+high-reso', 'SVM+', val, obsFuzz])
    print(t)
    ofile.write("SVMPlus\n")
    ofile.write("C = %f, gamma = %f, errRate = %f, eff = %f, val = %f, obsFuzz = %f \n" %
                (C, gamma, errRate, eff, val, obsFuzz))

    ofile.close()
    # pm.CalibrationPlot(pValues, y_test)
    # To be completed


    # pm.CalibrationPlot(pValues, y_test)

    # To be completed


# some settings before experiment
DATASET = MMP
svmFilename = phyChemFile[DATASET]
svmPlusFilename = morganUnhashedFiles[DATASET]
tunedKParam = tunedParam[DATASET][0]
tunedKStarParam = tunedParam[DATASET][1]

if __name__ == "__main__":
    ICPWithSVM("mnist", C = 100, gamma= .0001)
