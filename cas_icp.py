from fileUtils import csrFormat as csr
from fileUtils import csvFormat as csv
import os
from sklearn.model_selection import train_test_split
from conformalClassification import icp
from conformalClassification import perf_measure as pm
from fileUtils import parTuneWithCV as tune
from prettytable import PrettyTable
import numpy as np


svmFileName = "smiles_cas_N6512.sdf.std_class.sdf_descriptors.csv"

svmPlusFileName = "smiles_cas_N6512.sdf.std_class.sdf_SVMLIGHT_Morgan_unhashed_radius_3.csv"


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
def gridSearchWithCV(X_train, X_test,fileName):
     return tune.parSVMGridSearch(X_train, X_test, fileName)


# run SVM+ for sign descriptor files
def gridSearchSVMPlus(X_train, X_test, XStar_train,
                      kernelParam, kernelParamStar, logFile):
    return tune.gridSearchSVMPlus(X_train, X_test, XStar_train,
                                  kernelParam=kernelParam, kernelParamStar=kernelParamStar,
                                  logFile=logFile)


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

        #tuning for SVM on X
        C1, gamma1 = gridSearchWithCV(X_train, y_train, svmFile)

        C2, gamma2 = gridSearchWithCV(XStar_train, yStar_train, svmPlusFile)

        C, gamma = gridSearchSVMPlus(X_train, y_train, XStar_train, logFile=svmPlusFile,
                                     kernelParam= gamma1, kernelParamStar=gamma2 )

        #SVM on X
        pValues, y_predict = icp.ICPClassification(X_train, y_train, X_test,
                                                   C= C1, gamma=gamma1,
                                                   X_calib=X_calib, y_calib=y_calib)
        correct = np.sum(y_predict == y_test)
        accuracy = correct / len(y_predict)

        errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(pValues, y_test)
        avgSvmVal = avgSvmVal+val
        avgSvmOF = avgSvmOF+obsFuzz
        avgSvmAcc = avgSvmAcc + accuracy
        ofile.write("C = %f, gamma = %f, accuracy = %f \n" %
                    (C1, gamma1, accuracy))
        ofile.write("errRate = %f, eff = %f, val = %f, obsFuzz = %f \n" %
                    (errRate, eff, val, obsFuzz))
        ofile.close()

        ofile = open(dirPath + svmFile, "a")
        #y_test[y_test == 0] = -1

        # SVM on X Star
        pValues, y_predict = icp.ICPClassification(XStar_train, yStar_train, XStar_test,
                                                   C=C2, gamma=gamma2,
                                                   X_calib=XStar_calib, y_calib=yStar_calib)
        correct = np.sum(y_predict == y_test)
        accuracy = correct / len(y_predict)

        errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(pValues, y_test)
        avgSvmStarVal = avgSvmStarVal + val
        avgSvmStarOF = avgSvmStarOF + obsFuzz
        avgSvmStarAcc = avgSvmStarAcc + accuracy
        ofile.write("C = %f, gamma = %f, accuracy = %f \n" %
                    (C2, gamma2, accuracy))
        ofile.write("errRate = %f, eff = %f, val = %f, obsFuzz = %f \n" %
                    (errRate, eff, val, obsFuzz))
        ofile.close()
        ofile = open(dirPath + svmFile, "a")
        y_test[y_test == 0] = -1

        pValues, y_predict  = icp.ICPClassification(X_train, y_train, X_test,
                                        XStar=XStar_train, C=C, gamma=gamma,
                                        K=gamma1, KStar=gamma2,
                                        X_calib=X_calib, y_calib=y_calib)
        correct = np.sum(y_predict == y_test)
        accuracy = correct / len(y_predict)

        errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(pValues, y_test)
        avgSvmPlusVal = avgSvmPlusVal + val
        avgSvmPlusOF = avgSvmPlusOF + obsFuzz
        avgSvmPlusAcc = avgSvmPlusAcc + accuracy
        ofile.write("C = %f, gamma = %f, accuracy = %f \n" %
                    (C, gamma, accuracy))
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




