from fileUtils import csrFormat as csr
from fileUtils import csvFormat as csv
import os
from sklearn.model_selection import train_test_split
from conformalClassification import icp
from conformalClassification import perf_measure as pm
from fileUtils import tuneWithCV as tune
from prettytable import PrettyTable
import numpy as np


svmFileName = "smiles_cas_N6512.sdf.std_class.sdf_descriptors.csv"

svmPlusFileName = "smiles_cas_N6512.sdf.std_class.sdf_SVMLIGHT_Morgan_unhashed_radius_3.csv"

# Tuned values: 10, .01 : .915, 1, .01:914, 100,.01:915, 1000,.01,915
#
# tuned C and gamma kernel parameters
tunedSVMParam = [1000, .1]  # [10, .01]
tunedSVMStarParam = [1000, .01] #[1000,.01]
#tunedSVMPlus = [100, .001]
tunedSVMPlus = [100, .1]


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




