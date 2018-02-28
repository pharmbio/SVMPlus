# Special handling for MMP dataset, due to imbalance dataset
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
from fileUtils import tuneWithValidationSet as tune
from prettytable import PrettyTable

phyChemFile = "sr-mmp_nosalt.sdf.std_nodupl_class.sdf_descriptors.csv"

morganUnhashedFiles = "sr-mmp_nosalt.sdf.std_nodupl_class.sdf_SVMLIGHT_Morgan_unhashed_radius_3.csv"

# tuned kernel parameters
tunedSVMPhysChem = [10, .01]  # MMP

# tuned kernel parameters
tunedSVMMorgan = [100, .01]  # MMP, not yet tuned

tunedSVMPlus = [10, .01]  # MMP, not yet tuned


def prepareMMPDataset(fileName):
    path = str("MorganDataset/" + fileName)
    try:
        X, y = csr.readCSRFile(path)
    except:
        X, y = csv.loadDataset(path)

    dataClass1 = X[y == -1]
    # nClass1 = len(dataClass1)
    dataClass2 = X[y == 1]
    # nClass2 = len(dataClass2)

    # 500+500 for training/validation/testing
    nSample1 = 750
    nSample2 = 250
    testSize1 = 750
    testSize2 = 250
    X_train = np.vstack((dataClass1[:nSample1], dataClass2[:nSample2]))
    X_test = np.vstack((dataClass1[nSample1:(nSample1 + testSize1)], dataClass2[nSample2:(nSample2 + testSize2)]))
    X_valid = np.vstack((dataClass1[(nSample1 + testSize1):(nSample1 + 1500)],
                         dataClass2[(nSample2 + testSize2):(nSample2 + 500)]))
    y_train = np.concatenate((-np.ones(nSample1), np.ones(nSample2)))
    y_test = np.concatenate((-np.ones(testSize1), np.ones(testSize2)))
    y_valid = np.concatenate((-np.ones(testSize1), np.ones(testSize2)))

    return X_train, X_test, X_valid, y_train, y_test, y_valid


# Parameter estimation using grid search with validation set
def gridSearchWithValidation(fileName):
    X_train, X_test, X_valid, y_train, y_test, y_valid = prepareMMPDataset(fileName)
    tune.gridSearchWithValidation(X_train, X_test, X_valid, y_train, y_test, y_valid, fileName)


# run SVM+ for sign descriptor files
def gridSearchSVMPlus(svmFile, svmPlusFile,
                      kernelParam=0.0001, kernelParamStar=0.01):
    X_train, X_test, X_valid, y_train, y_test, y_valid = prepareMMPDataset(svmFile)
    XStar_train, XStar_test, XStar_valid, yStar_train, yStar_test, yStar_valid = prepareMMPDataset(svmFile)
    tune.gridSearchSVMPlus(X_train, X_test, X_valid, y_train, y_test, y_valid,
                           XStar_train, svmFile, kernelParam=kernelParam, kernelParamStar=kernelParamStar)


# run SVM for finger print descriptor file
def ICPWithSVM(svmFile, C=10, gamma=.01):
    X_train, X_test, X_valid, y_train, y_test, y_valid = prepareMMPDataset(svmFile)
    # record the results in a file
    dirPath = "icpWithSVMResults/"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    ofile = open(dirPath + svmFile, "a")
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
    X_train, X_test, X_valid, y_train, y_test, y_valid = prepareMMPDataset(svmFile)
    XStar_train, XStar_test, XStar_valid, yStar_train, yStar_test, yStar_valid = \
        prepareMMPDataset(svmPlusFile)

    dirPath = "icpWithSVMPlusResults/"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    ofile = open(dirPath + svmFile, "a")
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


if __name__ == "__main__":
    # readDetailsDescriptorFiles()

    # tune SVM parameters
    # for i in range(2):
    gridSearchWithValidation(phyChemFile)
    '''
    gridSearchWithValidation(phyChemFile)
    gridSearchWithValidation(morganUnhashedFiles)

    # tune SVM plus parameters
    gridSearchSVMPlus(phyChemFile, morganUnhashedFiles,
                      kernelParam=tunedSVMPhysChem[1],
                      kernelParamStar=tunedSVMMorgan[1])



    val, obsFuzz = ICPWithSVM(phyChemFile,
                              C=tunedSVMPhysChem[0], gamma=tunedSVMPhysChem[1])
    t = PrettyTable(['Dataset-BURSI', 'M/L Method', 'Validity', 'Obs-fuzziness'])
    t.add_row(['phys-chem', 'SVM', val, obsFuzz])
    val, obsFuzz = ICPWithSVM(morganUnhashedFiles, C=tunedSVMMorgan[0], gamma=tunedSVMMorgan[1])
    t.add_row(['morgan', 'SVM', val, obsFuzz])
    val, obsFuzz = ICPWithSVMPlus(phyChemFile, morganUnhashedFiles,
                                  C=tunedSVMPlus[0], gamma=tunedSVMPlus[1],
                                  kernelParam=tunedSVMPhysChem[1], kernelParamStar=tunedSVMMorgan[1])
    t.add_row(['phys-chem+morgan', 'SVM+', val, obsFuzz])
    print(t)
    '''

