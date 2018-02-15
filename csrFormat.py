import numpy as np
import svmPlusOpt as svmPlus
from cvxopt import matrix
import csv as csv
from sklearn.model_selection import train_test_split
from numpy.random import RandomState
import math
from sklearn import svm
from scipy.sparse import csr_matrix

def formatStr(nRow, strInput):
    p = strInput.find(":")
    colNo = strInput[:p]
    data = strInput[p+1:]
    return [nRow, colNo, data]


# to make sure there are no missing values
def readCSRFile():
    data = []
    # Read the dataset from given file
    file = open("DescriptorDataset/dataset_cas_N6512.csr")
    reader = csv.reader(file,  delimiter=' ')

    nRow = 0 # number of rows

    colList = []
    rowList = []
    dataList = []
    for row in reader:
        # skip label
        row = row[1:]
        newRow = [formatStr(nRow, val) if val else 0 for val in row]
        #print(newRow)
        rowList.append([x[0] for x in newRow])
        colList.append([x[1] for x in newRow])
        dataList.append([x[2] for x in newRow])
        nRow = nRow+1
    file.close()

    colums = np.array([x for x in colList[0]]).flatten()
    rows = np.array([x for x in rowList[0]]).flatten()
    data = np.array([x for x in dataList[0]]).flatten()
    for i in range(1,nRow):
        colums = np.append(colums, np.array([x for x in colList[i]]).flatten())
        rows = np.append(rows, np.array([x for x in rowList[i]]).flatten())
        data = np.append(data, np.array([x for x in dataList[i]]).flatten())

    print(len(colums))
    print(len(rows))
    print(len(data))
    print(colums[0:10])
    nFeatures = max(np.array(colums).astype(int))+1
    print(nFeatures)
    nSamples = nRow
    print(nRow)
    csMatrix = csr_matrix((np.array(data).astype(float), (np.array(rows).astype(int), np.array(colums).astype(int))))
    X = csMatrix.toarray()
    print(X.shape)
    print(X[len(X)-1, :100])

readCSRFile()