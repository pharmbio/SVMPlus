import numpy as np
import csv as csv
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from numpy.random import RandomState

def formatStr(strInput):
    p = strInput.find(":")
    colNo = strInput[:p]
    data = strInput[p+1:]
    if(colNo == "0"):
        print()
    return float(colNo), float(data)


# to make sure there are no missing values
def readCSRFile(fileName, split = False, returnIndices = False):
    # Read the dataset from given file
    file = open(fileName)
    reader = csv.reader(file, delimiter = '\t')
    nRow = 0 # number of rows
    colList = []
    rowList = []
    dataList = []
    labels = []
    for row in reader:
        # skip label
        labels = labels + row[:1]
        row = row[1:]
        for val in row:
            if val:
                colNo, data = formatStr(val)
                rowList.append(nRow)
                colList.append(colNo)
                dataList.append(data)
        nRow = nRow+1
    file.close()
    y = np.array([x for x in labels]).astype(int) #np.array(labels).astype(int).T
    columns = np.array(colList).astype(int)
    rows = np.array(rowList).astype(int)
    data = np.array(dataList)
    csMatrix = csr_matrix((np.array(data).astype(float), (np.array(rows).astype(int), np.array(columns).astype(int))))
    X = csMatrix.toarray()
    if split:
        if returnIndices:
            return train_test_split(X, y, range(nRow), test_size=0.4, random_state=RandomState())
        else:
            return train_test_split(X, y, test_size=0.4, random_state=RandomState())
    else:
        return X, y



if __name__ == "__main__":
    #X, y = readCSRFile("DescriptorDataset/bursi_nosalts_molsign.sdf.txt_SVMLIGHT_Morgan_unhashed_radius_3.csv")
    #X,y = readCSRFile("DescriptorDataset/smiles_cas_N6512.sdf.std_class.sdf_SVMLIGHT_Morgan_unhashed_radius_3.csv")
    X, y = readCSRFile("DescriptorDataset/sr-mmp_nosalt.sdf.std_nodupl_class.sdf_SVMLIGHT_Morgan_unhashed_radius_3.csv")
