import numpy as np
import svmPlusOpt as svmPlus
from cvxopt import matrix
import csv as csv
from sklearn.model_selection import train_test_split
from numpy.random import RandomState
import math
from sklearn import svm
import csrFormat as csr
import sys

# to make sure there are no missing values
def getIndicesWithMissingValues(fileName):
    data = []
    # Read the dataset from given file
    file = open(fileName)
    reader = csv.reader(file)
    indicesWithMissValues = []
    i=0
    for row in reader:
        newRow = [float(val) if val else math.nan for val in row]
        data.append(newRow)
    file.close()

    for x in data:
        if math.nan in x:
            indicesWithMissValues.append(i)
            i = i+1

    print("%d rows have missing values out of %d rows" % (len(indicesWithMissValues), len(data)))
    return indicesWithMissValues



# Load the dataset with given file name, and split it into train and test set
def loadDataset(fileName, split=False, returnIndices = False):
    data = []
    # Read the dataset
    file = open(fileName)
    reader = csv.reader(file)
    for row in reader:
        newRow = [float(val) if val else 0 for val in row]
        data.append(newRow)
    file.close()

    n = len(data) # number of observations
    X = np.array([x[1:] for x in data]).astype(float)
    y = np.array([x[0] for x in data]).astype(np.int) #labels

    del data # free up the memory

    if split:
        if returnIndices:
            return train_test_split(X, y, range(n), test_size=0.4, random_state=RandomState())
        else:
            return train_test_split(X, y, test_size=0.4, random_state=RandomState())
    else:
        return X, y


#run SVM for sign descriptor file
def runSVMSigDescFile(fileName, C = 10):
    paramC = np.linspace(1, 100, 20)
    paramGama = np.linspace(.1, .00001, 20)

    X_train, X_test, y_train, y_test =  loadDataset(fileName, split = True, returnIndices = False)
    ofile = open('paramGridResults/' + fileName[18:-4], "a")
    ofile.write("Size of the train set"+ str(X_train.shape[0])+ "\n")
    ofile.write("Size of the test set"+ str(X_test.shape[0])+ "\n")

    # fit svm model
    for i in range(20):
        for j in range(20):
            clf = svm.SVC(gamma = paramGama[j], C = paramC[i])
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            correct = np.sum(y_predict == y_test)
            print("Prediction accuracy using SVM for ", fileName)
            print("%d out of %d predictions correct" % (correct, len(y_predict)))
            predAcc = round(correct/len(y_predict), 3)
            ofile.write("param C = %f, gamma = %f, pred accuracy = %f \n" % (paramC[i],paramGama[j],predAcc))

    ofile.close()

#run SVM+ for sign descriptor files
def compareSVMAndSVMPlus(fileName1, fileName2, fingDescFile):
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
        sys.exit("different split for X and XStar")

    XFD, yFD = csr.readCSRFile(fingDescFile)

    XFD_train = XFD[indices_train]
    XFD_test = XFD[indices_test]
    yFD_train = yFD[indices_train]
    yFD_test = yFD[indices_test]

    if (y_train == yFD_train).all() and (y_test == yFD_test).all():
        print("same split for both the files")
    else:
        sys.exit("different split for X and XFD")

    # compute prediction accuracy using SVM for file1
    #svm_clf = svmPlus.svmPlusOpt(X_train, y_train, XStar=None, C=10, kernel="linear",
    #                             kernelParam=None)  # standard SVM
    #y_predict = svmPlus.predict(X_test, svm_clf)
    #correct = np.sum(y_predict == y_test)
    #print("Prediction accuracy using SVM for ", fileName1)
    #print("%d out of %d predictions correct" % (correct, len(y_predict)))

    clf = svm.SVC(gamma=.00001, C=100.)
    clf.fit(X_train, y_train)
    #print(len(clf.support_), "support vectors")
    y_predict = clf.predict(X_test[:1000])
    correct = np.sum(y_predict == y_test[:1000])
    print("Prediction accuracy using SVM for ", fileName1)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))
    
    # compute prediction accuracy using SVM for file1
    svm_clf = svmPlus.svmPlusOpt(XStar_train, yStar_train, XStar=None, C=100, kernel="rbf",
                                 kernelParam=.00001)  # standard SVM
    y_predict = svmPlus.predict(XStar_test[:1000], svm_clf)
    correct = np.sum(y_predict == yStar_test[:1000])
    print("Prediction accuracy using SVM for ", fileName2)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))

    # compute prediction accuracy using SVM for file1
    svm_clf = svmPlus.svmPlusOpt(XFD_train, yFD_train, XStar=None, C=100, kernel="rbf",
                                 kernelParam=.00001)  # standard SVM
    y_predict = svmPlus.predict(XFD_test[:1000], svm_clf)
    correct = np.sum(y_predict == yFD_test[:1000])
    print("Prediction accuracy using SVM for ", fingDescFile)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))

    # compute prediction accuracy using SVM+ for file1, and file2 as a priv-info
    clf = svmPlus.svmPlusOpt(X_train, y_train, XStar=XStar_train, C=100, kernel="rbf",
                             kernelParam=0.00001, kernelStar="rbf", kernelStarParam=0.00001,
                             gamma=0.0001) #linear kernel as of now
    y_predict = svmPlus.predict(X_test[:1000], clf)
    correct = np.sum(y_predict == y_test[:1000])
    print("Prediction accuracy using SVM+ for ", fileName1, fileName2)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))

    # compute prediction accuracy using SVM+ for file1, and file2 as a priv-info
    clf = svmPlus.svmPlusOpt(X_train, y_train, XStar=XFD_train, C=100, kernel="rbf",
                             kernelParam=0.00001, kernelStar="rbf", kernelStarParam=0.00001,
                             gamma=0.0001)  # linear kernel as of now
    y_predict = svmPlus.predict(X_test[:1000], clf)
    correct = np.sum(y_predict == y_test[:1000])
    print("Prediction accuracy using SVM+ for ", fileName1, fingDescFile)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))


# Print dimensions of the various descriptor files
def readDetailsDescriptorFiles():
    for fileName in listFileName:
        X, y = loadDataset(fileName)
        print("Details of the file", fileName)
        print(X.shape)
        getIndicesWithMissingValues(fileName) #print info about missing values

    for fileName in fingerPrintFiles:
        X, y = csr.readCSRFile(fileName)
        print("Details of the file", fileName)
        print(X.shape)


#global Variables: descriptor file names
listFileName = ["DescriptorDataset/sr-mmp_nosalt.sdf.std_nodupl_class.sdf_MACCS_166bit.csv",
    "DescriptorDataset/sr-mmp_nosalt.sdf.std_nodupl_class.sdf_topological.csv",
    "DescriptorDataset/smiles_cas_N6512.sdf.std_class.sdf_MACCS_166bit.csv",
    "DescriptorDataset/smiles_cas_N6512.sdf.std_class.sdf_topological.csv",
    "DescriptorDataset/bursi_nosalts_molsign.sdf.txt_MACCS_166bit.csv",
    "DescriptorDataset/bursi_nosalts_molsign.sdf.txt_topological.csv" ]

fingerPrintFiles = ["DescriptorDataset/sr-mmp_nosalt.sdf.std_nodupl_class.sdf_SVMLIGHT_Morgan_unhashed_radius_3.csv",
                    "DescriptorDataset/smiles_cas_N6512.sdf.std_class.sdf_SVMLIGHT_Morgan_unhashed_radius_3.csv",
                    "DescriptorDataset/bursi_nosalts_molsign.sdf.txt_SVMLIGHT_Morgan_unhashed_radius_3.csv"
                    ]


#run SVM for sign descriptor file
def svmOnFingDescFile(fileName, C = 10, gamma = .01):
    X_train, X_test, y_train, y_test =  csr.readCSRFile(fileName, split = True)
    # fit svm model
    svm_clf = svmPlus.svmPlusOpt(X_train, y_train, XStar=None, C = C, kernel="rbf",
                             kernelParam = gamma) # standard SVM
    y_predict = svmPlus.predict(X_test, svm_clf)
    correct = np.sum(y_predict == y_test)
    print("Prediction accuracy using SVM for ", fileName)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))


if __name__ == "__main__":
    #readDetailsDescriptorFiles()
    #for fileName in listFileName:
    #    runSVMSigDescFile(fileName)
    runSVMSigDescFile(listFileName[2])
    #svmOnFingDescFile(fingerPrintFiles[0])
    #compareSVMAndSVMPlus(listFileName[0], listFileName[1], fingerPrintFiles[0])
    #compareSVMAndSVMPlus(listFileName[2], listFileName[3], fingerPrintFiles[1])
    #compareSVMAndSVMPlus(listFileName[4], listFileName[5], fingerPrintFiles[2])


