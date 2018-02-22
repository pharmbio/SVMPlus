import numpy as np
import csv as csv
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#from numpy.random import RandomState
import math


# to make sure there are no missing values
def getIndicesWithMissingValues(fileName):
    data = []
    # Read the dataset from given file
    file = open(fileName)
    reader = csv.reader(file)
    next(reader)
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
    reader = csv.reader(file, delimiter = '\t')
    next(reader)
    for row in reader:
        newRow = [float(val) if val else 0 for val in row]
        data.append(newRow)
    file.close()

    n = len(data) # number of observations
    X = np.array([x[1:] for x in data]).astype(float)
    y = np.array([x[0] for x in data]).astype(np.int) #labels

    X = preprocessing.scale(X) # standardize the data
    del data # free up the memory

    if split:
        if returnIndices:
            return train_test_split(X, y, range(n), test_size=0.2, stratify = y, random_state = 7)
        else:
            return train_test_split(X, y, test_size=0.2, stratify = y, random_state = 7)
    else:
        return X, y

