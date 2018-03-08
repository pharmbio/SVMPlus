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

#Tuned values: 10, .01 : .915, 1, .01:914, 100,.01:915, 1000,.01,915
#
# tuned SVM C/kernel parameters
tunedSVMPhysChem = [1000, .1]# [10, .01]
tunedSVMMorgan = [1000, .001]
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
        train_test_split(X, y, range(len(X)), test_size=0.3,
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
    tune.gridSearchWithCV(X_train, y_train, fileName)


# run SVM+ for sign descriptor files
def gridSearchSVMPlus(svmFile, svmPlusFile, kernelParam, kernelParamStar):
    X_train, X_test, X_calib, y_train, y_test, y_calib, XStar_train = prepareDataset(svmFile, svmPlusFile)
    tune.gridSearchSVMPlus(X_train, y_train, XStar_train,
                           kernelParam=kernelParam, kernelParamStar=kernelParamStar, logFile=svmFile)


# some settings before experiment

if __name__ == "__main__":
    gridSearchWithCV(phyChemFile)
    #gridSearchWithCV(morganUnhashedFiles)
    '''
    # tune SVM/SVM+ parameters
    gridSearchWithValidation(phyChemFile)
    gridSearchWithValidation(morganUnhashedFiles)
    gridSearchSVMPlus(phyChemFile, morganUnhashedFiles,
                      kernelParam=tunedSVMPhysChem[1],
                      kernelParamStar=tunedSVMMorgan[1])
    '''
 


