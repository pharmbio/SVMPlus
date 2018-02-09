import numpy as np
import svmPlusOpt as svmPlus
from cvxopt import matrix
import csv as csv
from sklearn.model_selection import train_test_split
from numpy.random import RandomState


def loadDataset(fileName, split=False, returnIndices = False):
    data = []

    # Read the dataset from given file
    file = open(fileName)
    reader = csv.reader(file)
    for row in reader:
        data.append(row)
        p = len(row) # number of features+1 (labels)
    file.close()

    p = p-1 # number of features, minus the label
    n = len(data) # number of observations
    X = np.zeros((n,p))
    indices = range(n)
    try:
        X = np.array([x[1:] for x in data]).astype(float)
    except ValueError:
        # for missing values substitute zero
        XTemp = np.array([x[1:] for x in data])
        for i in range(0, n):
            for j in range(1, p):
                try:
                    X[i, j] = float(XTemp[i,j])
                except ValueError:
                    X[i, j] = 0
        del XTemp # free up the memory

    y = np.array([x[0] for x in data]).astype(np.int) #labels
    del data # free up the memory

    if split:
        if returnIndices:
            return train_test_split(X, y, indices, test_size=0.3, random_state = RandomState())
        else:
            return train_test_split(X, y, test_size=0.3, random_state=RandomState())
    else:
        return X, y


#run SVM for sign descriptor file
def runSVMSigDescFile(fileName, C = 10):
    X_train, X_test, y_train, y_test =  loadDataset(fileName, split = True, returnIndices = False)
    print(X_train.shape)
    print(X_test.shape)
    # fit svm model
    svm_clf = svmPlus.svmPlusOpt(X_train, y_train, XStar=None, C=C, kernel="rbf",
                             kernelParam=.5) # standard SVM

    y_predict = svmPlus.predict(X_test, svm_clf)
    correct = np.sum(y_predict == y_test)
    print("Prediction accuracy using SVM for ", fileName)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))

#run SVM+ for sign descriptor files
def compareSVMAndSVMPlus(fileName1, fileName2):
    X_train, X_test, y_train, y_test, indices_train,indices_test = \
        loadDataset(fileName1, split = True, returnIndices = True)

    XStar, yStar = loadDataset(fileName2, split=False)

    XStar_train = XStar[indices_train]
    XStar_test = XStar[indices_test]
    yStar_train = yStar[indices_train]
    yStar_test = yStar[indices_test]

    print(X_train.shape)
    print(XStar_train.shape)

    if (y_train == yStar_train).all() and (y_test == yStar_test).all() :
        print("same split for both the files")
    else:
        print("different split for both the files")

    # compute prediction accuracy using SVM for file1
    svm_clf = svmPlus.svmPlusOpt(X_train, y_train, XStar=None, C=10, kernel="linear",
                                 kernelParam=None)  # standard SVM
    y_predict = svmPlus.predict(X_test, svm_clf)
    correct = np.sum(y_predict == y_test)
    print("Prediction accuracy using SVM for ", fileName1)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))

    # compute prediction accuracy using SVM for file1
    svm_clf = svmPlus.svmPlusOpt(XStar_train, yStar_train, XStar=None, C=10, kernel="linear",
                                 kernelParam=None)  # standard SVM
    y_predict = svmPlus.predict(XStar_test, svm_clf)
    correct = np.sum(y_predict == yStar_test)
    print("Prediction accuracy using SVM for ", fileName2)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))

    # compute prediction accuracy using SVM+ for file1, and file2 as a priv-info
    clf = svmPlus.svmPlusOpt(X_train, y_train, XStar=XStar_train, C=10, kernel="linear",
                             kernelParam=None, kernelStar="linear", kernelStarParam=None,
                             gamma=1) #linear kernel as of now
    y_predict = svmPlus.predict(X_test, clf)
    correct = np.sum(y_predict == y_test)
    print("Prediction accuracy using SVM+ for ", fileName1, fileName2)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))



# Print dimensions of the vaious descriptor files
def readDetailsDescriptorFiles():
    for fileName in listFileName:
        X, y = loadDataset(fileName)
        print("Details of the file", fileName)
        print(X.shape)



#global Variables: descriptor file names
listFileName = ["DescriptorDataset/sr-mmp_nosalt.sdf.std_nodupl_class.sdf_toplogical.csv",
    "DescriptorDataset/sr-mmp_nosalt.sdf.std_nodupl_class.sdf_MACCS_166bit.csv",
    "DescriptorDataset/smiles_cas_N6512.sdf.std_class.sdf_toplogical.csv",
    "DescriptorDataset/smiles_cas_N6512.sdf.std_class.sdf_MACCS_166bit.csv",
    "DescriptorDataset/bursi_nosalts_molsign.sdf.txt_MACCS_166bit.csv",
    "DescriptorDataset/bursi_nosalts_molsign.sdf.txt_toplogical.csv" ]


if __name__ == "__main__":
    #readDetailsDescriptorFiles()
    #for fileName in listFileName:
    #    runSVMSigDescFile(fileName)
    #compareSVMAndSVMPlus(listFileName[1], listFileName[0])
    runSVMSigDescFile(listFileName[1], C = 100)

