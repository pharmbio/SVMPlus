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


phyChemFile = ["bursi_nosalts_molsign.sdf.txt_descriptors.csv",
               "smiles_cas_N6512.sdf.std_class.sdf_descriptors.csv",
               "sr-mmp_nosalt.sdf.std_nodupl_class.sdf_descriptors.csv"]

morganUnhashedFiles = ["bursi_nosalts_molsign.sdf.txt_SVMLIGHT_Morgan_unhashed_radius_3.csv",
                       "smiles_cas_N6512.sdf.std_class.sdf_SVMLIGHT_Morgan_unhashed_radius_3.csv",
                       "sr-mmp_nosalt.sdf.std_nodupl_class.sdf_SVMLIGHT_Morgan_unhashed_radius_3.csv"]

PHYSCHEM = 0
UNHASHED = 1

BURSI = 0
CAS = 1
MMP = 2

# tuned kernel parameters
tunedSVMPhysChem = [ [1000, .01], #[100, .00001], #[10, .1],  # BURSI: or 1, .1
              [1000, .1],  # CAS or 100 .01
              [10, .01]]  # MMP


# tuned kernel parameters
tunedSVMMorgan = [[10, .001], #1000, .0001
              [1,.01], #10, .01
              [100, .01]] # MMP, not yet tuned

tunedSVMPlus = [[10, .01],#[1, .001], #BURSI
                [1000, .01], #[10, .001], # CAS
                [10, .001]]  # MMP, not yet tuned

def prepareDataset(fileName):
    path = str("MorganDataset/"+fileName)
    try:
        X, y = csr.readCSRFile(path)
    except:
        X, y = csv.loadDataset(path)

    dataClass1 = X[y == -1]
    # nClass1 = len(dataClass1)
    dataClass2 = X[y == 1]
    # nClass2 = len(dataClass2)

    # 500+500 for training/validation/testing
    nSample = 500
    testSize = 500
    validSize = 500
    X_train = np.vstack((dataClass1[:nSample], dataClass2[:nSample]))
    X_test = np.vstack((dataClass1[nSample:(nSample + testSize)], dataClass2[nSample:(nSample + testSize)]))
    X_valid = np.vstack((dataClass1[(nSample + testSize):(nSample + 1000)],
                         dataClass2[(nSample + testSize):(nSample + 1000)]))
    y_train = np.concatenate((-np.ones(nSample), np.ones(nSample)))
    y_test = np.concatenate((-np.ones(testSize), np.ones(testSize)))
    y_valid = np.concatenate((-np.ones(validSize), np.ones(validSize)))

    return X_train, X_test, X_valid, y_train, y_test, y_valid

# Parameter estimation using grid search with validation set
def gridSearchWithValidation(fileName):
    X_train, X_test, X_valid, y_train, y_test, y_valid = prepareDataset(fileName)
    tune.gridSearchWithValidation(X_train, X_test, X_valid, y_train, y_test, y_valid, fileName)


# run SVM+ for sign descriptor files
def gridSearchSVMPlus(svmFile, svmPlusFile,
                      kernelParam=0.0001, kernelParamStar=0.01):
    X_train, X_test, X_valid, y_train, y_test, y_valid = prepareDataset(svmFile)
    XStar_train, XStar_test, XStar_valid, yStar_train, yStar_test, yStar_valid = prepareDataset(svmFile)
    tune.gridSearchSVMPlus(X_train, X_test, X_valid, y_train, y_test, y_valid,
                           XStar_train, svmFile, kernelParam= kernelParam, kernelParamStar= kernelParamStar)


# run SVM for finger print descriptor file
def ICPWithSVM(svmFile, C=10, gamma=.01):
    X_train, X_test, X_valid, y_train, y_test, y_valid = prepareDataset(svmFile)
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
    #pm.CalibrationPlot(pValues, y_test)
    # To be completed



# run SVM Plus for finger print descriptor file
def ICPWithSVMPlus(svmFile, svmPlusFile, C=10, gamma=.01,
                   kernelParam=0.01, kernelParamStar=0.01):
    X_train, X_test, X_valid, y_train, y_test, y_valid = prepareDataset(svmFile)
    XStar_train, XStar_test, XStar_valid, yStar_train, yStar_test, yStar_valid = \
        prepareDataset(svmPlusFile)

    dirPath = "icpWithSVMPlusResults/"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    ofile = open(dirPath + svmFile, "a")
    ofile.write("Size of the train set: " + str(X_train.shape[0]) + "\n")
    ofile.write("Size of the test set: " + str(X_test.shape[0]) + "\n")

    pValues = icp.ICPClassification(X_train, y_train, X_test,
                                    XStar = XStar_train, C = C, gamma= gamma,
                                    K = kernelParam, KStar= kernelParamStar,
                                    X_calib=X_valid, y_calib=y_valid)
    y_test = y_test.astype(int)
    y_test[y_test == -1] = 0

    errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(pValues, y_test)
    # open again
    ofile.write("C = %f, gamma = %f, errRate = %f, eff = %f, val = %f, obsFuzz = %f \n" %
                (C, gamma, errRate, eff, val, obsFuzz))

    ofile.close()
    return val, obsFuzz
    #pm.CalibrationPlot(pValues, y_test)

    # To be completed



# some settings before experiment
DATASET = CAS
svmFilename = phyChemFile[DATASET]
svmPlusFilename = morganUnhashedFiles[DATASET]
tunedKParam = tunedSVMPhysChem[DATASET][1]
tunedKStarParam = tunedSVMMorgan[DATASET][1]

if __name__ == "__main__":
    # readDetailsDescriptorFiles()

    #tune SVM parameters
    #for i in range(2):
    #gridSearchWithValidation(phyChemFile[DATASET])
   
    #gridSearchSVMPlus(phyChemFile[DATASET], morganUnhashedFiles[DATASET],
    #                  kernelParam=tunedSVMPhysChem[DATASET][1],
    #                  kernelParamStar=tunedSVMMorgan[DATASET][1])

    #gridSearchWithValidation(phyChemFile[2])
    #gridSearchWithValidation(morganUnhashedFiles[2])
    '''
    val, obsFuzz = ICPWithSVM(svmFilename,
                              C=1000, gamma=.01)
    t = PrettyTable(['Dataset-BURSI', 'M/L Method', 'Validity', 'Obs-fuzziness'])
    t.add_row(['phys-chem', 'SVM', val, obsFuzz])
    print(t)
    
    # tune SVM plus parameters
    for i in range(2):
        gridSearchSVMPlus(phyChemFile[i], morganUnhashedFiles[i],
                          kernelParam=tunedSVMPhysChem[i][1], 
                          kernelParamStar=tunedSVMMorgan[i][1])

    
    '''

    val, obsFuzz = ICPWithSVM(svmFilename,
                              C=tunedSVMPhysChem[DATASET][0], gamma=tunedSVMPhysChem[DATASET][1])
    t = PrettyTable(['Dataset-BURSI', 'M/L Method', 'Validity', 'Obs-fuzziness'])
    t.add_row(['phys-chem', 'SVM', val, obsFuzz])

    
    val, obsFuzz = ICPWithSVM(svmPlusFilename, C=tunedSVMMorgan[DATASET][0], gamma=tunedSVMMorgan[DATASET][1])
    t.add_row(['morgan', 'SVM', val, obsFuzz])
    val, obsFuzz = ICPWithSVMPlus(svmFilename, svmPlusFilename,
                C=tunedSVMPlus[DATASET][0], gamma=tunedSVMPlus[DATASET][1],
                kernelParam=tunedSVMPhysChem[DATASET][1], kernelParamStar=tunedSVMMorgan[DATASET][1])
    t.add_row(['phys-chem+morgan', 'SVM+', val, obsFuzz])

    print(t)


