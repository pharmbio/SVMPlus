from fileUtils import csrFormat as csr
from fileUtils import csvFormat as csv
import os
from sklearn.model_selection import train_test_split
from conformalClassification import icp
from conformalClassification import perf_measure as pm
from fileUtils import tuneWithCV as tune
from prettytable import PrettyTable
import numpy as np


svmFileName = "sr-mmp_nosalt.sdf.std_nodupl_class.sdf_descriptors.csv"

svmPlusFileName = "sr-mmp_nosalt.sdf.std_nodupl_class.sdf_SVMLIGHT_Morgan_unhashed_radius_3.csv"

# Tuned values: 10, .01 : .915, 1, .01:914, 100,.01:915, 1000,.01,915
#
# tuned C and gamma kernel parameters
tunedSVMParam = [100, .01]  # [10, .01]
tunedSVMStarParam = [1000, .01]
tunedSVMPlus = [100, .001]


def prepareDataset(svmFile, svmPlusFile=None, returnAll = False):
    path = str("MorganDataset/" + svmFile)
    try:
        X, X_test, y, y_test, indices_x, indices_test = \
            csr.readCSRFile(path, split=True, returnIndices=True)
    except:
        X, X_test, y, y_test, indices_x, indices_test = \
            csv.loadDataset(path, split=True, returnIndices=True)

    X_train, X_calib, y_train, y_calib, indices_train, indices_valid = \
        train_test_split(X, y, range(len(X)), test_size=0.3)

    if svmPlusFile is not None:
        XStar, yStar = csr.readCSRFile(str("MorganDataset/" + svmPlusFile))
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
    X_train, X_test, X_calib, y_train, y_test, y_calib, \
    XStar_train, XStar_test, XStar_calib, yStar_train, yStar_test, yStar_calib   = \
        prepareDataset(svmFile, svmPlusFile, returnAll = True)

    dirPath = "icpResults/"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    avgSvmVal = 0
    avgSvmOF = 0
    avgSvmStarVal = 0
    avgSvmStarOF = 0
    avgSvmPlusVal = 0
    avgSvmPlusOF = 0
    for i in range(5):
        ofile = open(dirPath + svmFile, "a")
        ofile.write("Size of the train set: " + str(X_train.shape[0]) + "\n")
        ofile.write("Size of the test set: " + str(X_test.shape[0]) + "\n")

        #SVM on X
        pValues, y_predict = icp.ICPClassification(X_train, y_train, X_test,
                                                   C= tunedSVMParam[0], gamma=tunedSVMParam[1],
                                                   X_calib=X_calib, y_calib=y_calib)
        correct = np.sum(y_predict == y_test)
        accuracy = correct / len(y_predict)

        y_test = y_test.astype(int)
        y_test[y_test == -1] = 0

        errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(pValues, y_test)
        avgSvmVal = avgSvmVal+val
        avgSvmOF = avgSvmOF+obsFuzz
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

        y_test = y_test.astype(int)
        y_test[y_test == -1] = 0
        errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(pValues, y_test)
        avgSvmStarVal = avgSvmStarVal + val
        avgSvmStarOF = avgSvmStarOF + obsFuzz
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

        y_test = y_test.astype(int)
        y_test[y_test == -1] = 0

        errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(pValues, y_test)
        avgSvmPlusVal = avgSvmPlusVal + val
        avgSvmPlusOF = avgSvmPlusOF + obsFuzz
        ofile.write("C = %f, gamma = %f, accuracy = %f \n" %
                    (tunedSVMPlus[0], tunedSVMPlus[1], accuracy))
        ofile.write("errRate = %f, eff = %f, val = %f, obsFuzz = %f \n" %
                    (errRate, eff, val, obsFuzz))

        ofile.close()

    ofile = open(dirPath + svmFile, "a")
    ofile.write("svmVal = %f, svmOf = %f \n" %
                (avgSvmVal/5, avgSvmOF/5))
    ofile.write("svmVal = %f, svmOf = %f \n" %
                (avgSvmStarVal / 5, avgSvmStarOF / 5))
    ofile.write("svmVal = %f, svmOf = %f \n" %
                (avgSvmPlusVal / 5, avgSvmPlusOF / 5))
    ofile.close()
    return val, obsFuzz
    # pm.CalibrationPlot(pValues, y_test)





if __name__ == "__main__":
    # gridSearchWithCV(svmFileName)
    #gridSearchWithCV(svmPlusFileName)
    '''
    # tune SVM/SVM+ parameters
    gridSearchWithValidation(svmFileName)
    gridSearchWithValidation(svmPlusFileName)
    gridSearchSVMPlus(svmFileName, svmPlusFileName,
                      kernelParam=tunedSVMParam[1],
                      kernelParamStar=tunedSVMStarParam[1])

    '''
    ICPWithSVMAndSVMPlus(svmFileName, svmPlusFileName)

if 0:
    val, obsFuzz = ICPWithSVM(svmFileName,
                              C=tunedSVMParam[0], gamma=tunedSVMParam[1])
    t = PrettyTable(['Dataset-BURSI', 'M/L Method', 'Validity', 'Obs-fuzziness'])
    t.add_row(['phys-chem', 'SVM', val, obsFuzz])

    val, obsFuzz = ICPWithSVM(svmPlusFileName, C=tunedSVMStarParam[0], gamma=tunedSVMStarParam[1])
    t.add_row(['morgan', 'SVM', val, obsFuzz])
    val, obsFuzz = ICPWithSVMPlus(svmFileName, svmPlusFileName,
                                  C=tunedSVMPlus[0], gamma=tunedSVMPlus[1],
                                  kernelParam=tunedSVMParam[1], kernelParamStar=tunedSVMStarParam[1])
    t.add_row(['phys-chem+morgan', 'SVM+', val, obsFuzz])

    print(t)


