import numpy as np
import svmPlusOpt as svmPlus
from cvxopt import matrix
import csv as csv
from sklearn.model_selection import train_test_split
from numpy.random import RandomState
import math
from sklearn import svm

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

    x =.33333
    x = round(x,3)
    print(x)
    #print(X_train.shape)
    #print(X_test.shape)
    # fit svm model
    for i in range(20):
        for j in range(20):
            svm_clf = svmPlus.svmPlusOpt(X_train, y_train, XStar=None, C = paramC[i], kernel="rbf",
                                     kernelParam = paramGama[j]) # standard SVM
            y_predict = svmPlus.predict(X_test, svm_clf)
            correct = np.sum(y_predict == y_test)
            print("Prediction accuracy using SVM for ", fileName)
            print("%d out of %d predictions correct" % (correct, len(y_predict)))
            predAcc = round(correct/len(y_predict), 3)
            ofile.write("param C = %f, gamma = %f, pred accuracy = %f \n" % (paramC[i],paramGama[j],predAcc))

    ofile.close()

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
    #svm_clf = svmPlus.svmPlusOpt(X_train, y_train, XStar=None, C=10, kernel="linear",
    #                             kernelParam=None)  # standard SVM
    #y_predict = svmPlus.predict(X_test, svm_clf)
    #correct = np.sum(y_predict == y_test)
    #print("Prediction accuracy using SVM for ", fileName1)
    #print("%d out of %d predictions correct" % (correct, len(y_predict)))

    clf = svm.SVC(gamma=.00001, C=100.)
    clf.fit(X_train, y_train)
    print(len(clf.support_), "support vectors")
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

    # compute prediction accuracy using SVM+ for file1, and file2 as a priv-info
    clf = svmPlus.svmPlusOpt(X_train, y_train, XStar=XStar_train, C=100, kernel="rbf",
                             kernelParam=0.00001, kernelStar="rbf", kernelStarParam=0.00001,
                             gamma=0.0001) #linear kernel as of now
    y_predict = svmPlus.predict(X_test[:1000], clf)
    correct = np.sum(y_predict == y_test[:1000])
    print("Prediction accuracy using SVM+ for ", fileName1, fileName2)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))



# Print dimensions of the vaious descriptor files
def readDetailsDescriptorFiles():
    for fileName in listFileName:
        X, y = loadDataset(fileName)
        print("Details of the file", fileName)
        print(X.shape)
        getIndicesWithMissingValues(fileName) #print info about missing values



#global Variables: descriptor file names
listFileName = ["DescriptorDataset/sr-mmp_nosalt.sdf.std_nodupl_class.sdf_MACCS_166bit.csv",
    "DescriptorDataset/sr-mmp_nosalt.sdf.std_nodupl_class.sdf_topological.csv",
    "DescriptorDataset/smiles_cas_N6512.sdf.std_class.sdf_MACCS_166bit.csv",
    "DescriptorDataset/smiles_cas_N6512.sdf.std_class.sdf_topological.csv",
    "DescriptorDataset/bursi_nosalts_molsign.sdf.txt_MACCS_166bit.csv",
    "DescriptorDataset/bursi_nosalts_molsign.sdf.txt_topological.csv" ]


if __name__ == "__main__":
    #readDetailsDescriptorFiles()
    #for fileName in listFileName:
    #    runSVMSigDescFile(fileName)
    runSVMSigDescFile(listFileName[0])
    #compareSVMAndSVMPlus(listFileName[0], listFileName[1])
    #compareSVMAndSVMPlus(listFileName[2], listFileName[3])
    #compareSVMAndSVMPlus(listFileName[4], listFileName[5])


