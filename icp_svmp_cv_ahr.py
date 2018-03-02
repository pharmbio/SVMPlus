from fileUtils import csrFormat as csr
from fileUtils import csvFormat as csv
import os
from sklearn.model_selection import train_test_split
from conformalClassification import icp
from conformalClassification import perf_measure as pm
from fileUtils import tuneWithCV as tune
from prettytable import PrettyTable

phyChemFile = "nr-ahr_nosalt.sdf.std_nodupl_class.sdf_descriptors.csv"

morganUnhashedFiles = "nr-ahr_nosalt.sdf.std_nodupl_class.sdf_SVMLIGHT_UNHASHED_radius_3.csv"

# tuned SVM C/kernel parameters
tunedSVMPhysChem = [1000, .1]# [10, .01]
tunedSVMMorgan = [10, .01]
tunedSVMPlus = [100, .001]

def prepareDataset(svmFile, svmPlusFile = None):
    path = str("MorganDataset/" + svmFile)
    try:
        X, X_test, y, y_test, indices_x, indices_test = \
            csr.readCSRFile(path, split=True, returnIndices = True)
    except:
        X, X_test, y, y_test, indices_x, indices_test = \
            csv.loadDataset(path, split=True, returnIndices = True)


    X_train, X_calib, y_train, y_calib, indices_train, indices_valid = \
        train_test_split(X, y, range(len(X)), test_size=0.2,
                         stratify=y, random_state=7)

    if svmPlusFile is not None:
        XStar, yStar = csr.readCSRFile(str("MorganDataset/" + svmPlusFile))
        XStar = XStar[indices_x]
        XStar_train = XStar[indices_train]
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
    pValues = icp.ICPClassification(X_train, y_train, X_test,
                                    X_calib=X_calib, y_calib=y_calib)
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
    X_train, X_test, X_calib, y_train, y_test, y_calib, XStar_train = \
        prepareDataset(svmFile, svmPlusFile)

    dirPath = "icpWithSVMPlusResults/"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    ofile = open(dirPath + svmFile, "a")
    ofile.write("Size of the train set: " + str(X_train.shape[0]) + "\n")
    ofile.write("Size of the test set: " + str(X_test.shape[0]) + "\n")

    pValues = icp.ICPClassification(X_train, y_train, X_test,
                                    XStar=XStar_train, C=C, gamma=gamma,
                                    K=kernelParam, KStar=kernelParamStar,
                                    X_calib=X_calib, y_calib=y_calib)
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
    #gridSearchWithCV(phyChemFile)
    gridSearchWithCV(morganUnhashedFiles)
    '''
    # tune SVM/SVM+ parameters
    gridSearchWithValidation(phyChemFile)
    gridSearchWithValidation(morganUnhashedFiles)
    gridSearchSVMPlus(phyChemFile, morganUnhashedFiles,
                      kernelParam=tunedSVMPhysChem[1],
                      kernelParamStar=tunedSVMMorgan[1])

    '''
    if 0:
        val, obsFuzz = ICPWithSVM(phyChemFile,
                                  C=10, gamma=.1)
        t = PrettyTable(['Dataset-BURSI', 'M/L Method', 'Validity', 'Obs-fuzziness'])
        t.add_row(['phys-chem', 'SVM', val, obsFuzz])
        print(t)
        
    if 0:
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


