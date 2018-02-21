import numpy as np
import svmPlusOpt as svmPlus
from sklearn import svm
from fileUtils import csrFormat as csr
from fileUtils import csvFormat as csv
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import os


descriptorFiles = ["bursi_nosalts_molsign.sdf.txt_descriptors.csv",
                   "sr-mmp_nosalt.sdf.std_nodupl_class.sdf_descriptors.csv",
                   "smiles_cas_N6512.sdf.std_class.sdf_descriptors.csv"]
morgan512BitsFiles = ["bursi_nosalts_molsign.sdf.txt_SVMLIGHT_Morgan_512_bits_radius_3.csv",
                      "sr-mmp_nosalt.sdf.std_nodupl_class.sdf_SVMLIGHT_Morgan_512_bits_radius_3.csv",
                      "smiles_cas_N6512.sdf.std_class.sdf_SVMLIGHT_Morgan_512_bits_radius_3.csv"]

morgan512BitsUnhashedFiles = ["bursi_nosalts_molsign.sdf.txt_SVMLIGHT_Morgan_unhashed_radius_3.csv",
                      "sr-mmp_nosalt.sdf.std_nodupl_class.sdf_SVMLIGHT_Morgan_unhashed_radius_3.csv",
                      "smiles_cas_N6512.sdf.std_class.sdf_SVMLIGHT_Morgan_unhashed_radius_3.csv"]

morgan64BitsFiles = ["bursi_nosalts_molsign.sdf.txt_SVMLIGHT_Morgan_64_bits_radius_3.csv",
                     "sr-mmp_nosalt.sdf.std_nodupl_class.sdf_SVMLIGHT_Morgan_64_bits_radius_3.csv",
                     "smiles_cas_N6512.sdf.std_class.sdf_SVMLIGHT_Morgan_64_bits_radius_3.csv"]



# Parameter estimation using grid search with cross-validation
def gridSearchWithCV(fileName):
    paramC = [.001, .01, .1, 1, 100, 1000]
    paramGamma = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, .1]

    #X, X_test, y, y_test =  loadDataset(fileName, split = True) # train(80):test(20) split
    path = str("MorganDataset/"+fileName)
    try:
        X, X_test, y, y_test = csr.readCSRFile(path, split=True)
    except:
        X, X_test, y, y_test = csv.loadDataset(path, split=True)

    cv = StratifiedKFold(n_splits = 5)
    folds = [[train_index, test_index] for train_index, test_index in cv.split(X, y)]
    #record the results in a file
    dirPath = "paramGridResults/"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    ofile = open('paramGridResults/' + fileName, "a")
    ofile.write("Size of the train set: "+ str(X.shape[0])+ "\n")
    ofile.write("Size of the test set: "+ str(X_test.shape[0])+ "\n")
    ofile.close()
    #open again
    ofile = open('paramGridResults/' + fileName, "a")
    predAcc = []
    index = []
    rocAUC = []
    # Cross validate
    for i in range(len(paramC)):
        for j in range(len(paramGamma)):
            accuracy = 0
            areaUnderCurve = 0
            for fold in folds:
                clf = svm.SVC(gamma = paramGamma[j], C = paramC[i], probability=True)
                y_prob = clf.fit(X[fold[0]], y[fold[0]]).predict_proba(X[fold[1]])
                fpr, tpr, thresholds = roc_curve(y[fold[1]], y_prob[:, 1])
                y_predict = clf.predict(X[fold[1]])
                correct = np.sum(y_predict == y[fold[1]])
                accuracy = accuracy + (correct/len(y_predict))
                areaUnderCurve = areaUnderCurve + auc(fpr, tpr)

            print(i,j)
            accuracy = accuracy / 5
            areaUnderCurve= areaUnderCurve / 5
            rocAUC.append(areaUnderCurve)
            predAcc.append(accuracy)
            index.append([i,j])
            ofile.write("param C = %f, gamma = %f, mean pred accuracy = %f, mean AUC = %f \n" %
                        (paramC[i], paramGamma[j], accuracy, areaUnderCurve))
    selectedIndex = np.argmax(predAcc)
    C = paramC[index[selectedIndex][0]]
    gamma = paramGamma[index[selectedIndex][1]]

    #Fit the SVM on whole training set and calculate results on test set
    clf = svm.SVC(gamma=gamma, C=C)
    clf.fit(X, y)
    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)
    predAcc = round(correct / len(y_predict), 3)
    print("param C = %f, gamma = %f, pred accuracy = %f \n" % (C, gamma, predAcc))
    ofile.write("param C = %f, gamma = %f, pred accuracy = %f \n" % (C, gamma, predAcc))

    # select model based on AUC
    selectedIndex = np.argmax(rocAUC)
    C = paramC[index[selectedIndex][0]]
    gamma = paramGamma[index[selectedIndex][1]]

    # Fit the SVM on whole training set and calculate AUC for test set
    clf = svm.SVC(gamma=gamma, C=C, probability=True)
    y_prob = clf.fit(X, y).predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    finalAUC = auc(fpr, tpr)
    #print("param C = %f, gamma = %f, AUC = %f \n" % (C, gamma, finalAUC))
    ofile.write("param C = %f, gamma = %f, AUC = %f \n" % (C, gamma, finalAUC))

    ofile.close()



# run SVM+ for sign descriptor files
def gridSearchSVMPlus(svmFile, svmPlusFile,
                      kernelParam = 0.0001, kernelParamStar = 0.01):
    paramC = [.01, .1, 1, 100]
    paramGamma = [1e-4, 1e-3, 1e-2, .1]
    path = str("MorganDataset/" + svmFile)
    try:
        X, X_test, y, y_test, indices_train, indices_test = \
            csr.readCSRFile(path, split=True, returnIndices=True)
    except:
        X, X_test, y, y_test, indices_train, indices_test = \
            csv.loadDataset(path, split = True, returnIndices = True)

    path = str("MorganDataset/" + svmPlusFile)
    XStar, yStar = csr.readCSRFile(path)

    XStar_test = XStar[indices_test]
    XStar = XStar[indices_train]
    yStar_test = yStar[indices_test]
    yStar = yStar[indices_train]


    if (y == yStar).all() and (y_test == yStar_test).all() :
        print("same split for both the files")
    else:
        sys.exit("different split for X and XStar")

    # record the results in a file
    cv = StratifiedKFold(n_splits=5)
    folds = [[train_index, test_index] for train_index, test_index in cv.split(X, y)]
    predAcc = []
    index = []
    dirPath = "paramGridResults/"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    ofile = open('paramGridResults/' + "SVMPlus_" + svmFile, "a")
    ofile.write("Size of the train set: " + str(X.shape[0]) + "\n")
    ofile.write("Size of the test set: " + str(X_test.shape[0]) + "\n")
    ofile.close() #store file size and close, then open again
    ofile = open('paramGridResults/' + "SVMPlus_" + svmFile, "a")
    for i in range(len(paramC)):
        for j in range(len(paramGamma)):
            accuracy = 0
            for fold in folds:
                train_index = fold[0]
                test_index = fold[1]
                # compute prediction accuracy using SVM+ on svmFile, and svmPlusFile2 as a priv-info
                clf = svmPlus.svmPlusOpt(X[train_index], y[train_index], XStar=XStar[train_index],
                                         C = paramC[i], gamma = paramGamma[j],
                                         kernel="rbf", kernelParam=kernelParam,
                                         kernelStar="rbf", kernelStarParam=kernelParamStar)
                y_predict = svmPlus.predict(X[test_index], clf)
                correct = np.sum(y_predict == y[test_index])
                accuracy = accuracy + (correct / len(y_predict))

            print(i, j)
            accuracy = accuracy / 5
            predAcc.append(accuracy)
            index.append([i, j])
            ofile.write("param C = %f, gamma = %f, mean pred accuracy = %f \n" %
                        (paramC[i], paramGamma[j], accuracy))

    selectedIndex = np.argmax(predAcc)
    C = paramC[index[selectedIndex][0]]
    gamma = paramGamma[index[selectedIndex][1]]

    # For optimal parameters
    clf = svmPlus.svmPlusOpt(X, y, XStar= XStar, C=C, gamma=gamma,
                             kernel="rbf", kernelParam = kernelParam,
                             kernelStar="rbf", kernelStarParam = kernelParamStar)
    y_predict = svmPlus.predict(X_test, clf)
    correct = np.sum(y_predict == y_test)
    predAcc = round(correct / len(y_predict), 3)
    # print("param C = %f, gamma = %f, pred accuracy = %f \n" % (C, gamma, predAcc))
    #print("param C = %f, gamma = %f, AUC = %f \n" % (C, gamma, predAcc))
    ofile.write("param C = %f, gamma = %f, AUC = %f \n" % (C, gamma, predAcc))
    ofile.close()




# Print details of the various Morgan descriptor/FP files
def readDetailsDescriptorFiles():
    dirPath = "Results/"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    ofile = open(dirPath + "fileDetails.txt", "w")

    for fileName in descriptorFiles:
        path = str("MorganDataset/" + fileName)
        X, y = csv.loadDataset(path)
        ofile.write("fileName: %s \n" % (fileName))
        ofile.write("Dimension of the file %d X %d \n" % (X.shape))
        print("Details of the file:", fileName)
        print(X.shape)

    for fileName in morgan64BitsFiles:
        path = str("MorganDataset/" + fileName)
        X, y = csr.readCSRFile(path)
        ofile.write("fileName: %s \n" % (fileName))
        ofile.write("Dimension of the file %d X %d \n" % (X.shape))
        print("Details of the file:", fileName)
        print(X.shape)

    for fileName in morgan512BitsFiles :
        path = str("MorganDataset/" + fileName)
        X, y = csr.readCSRFile(path)
        ofile.write("fileName: %s \n" % (fileName))
        ofile.write("Dimension of the file %d X %d \n" % (X.shape))
        print("Details of the file:", fileName)
        print(X.shape)

    for fileName in morgan512BitsUnhashedFiles:
        path = str("MorganDataset/" + fileName)
        X, y = csr.readCSRFile(path)
        ofile.write("fileName: %s \n" % (fileName))
        ofile.write("Dimension of the file %d X %d \n" % (X.shape))
        print("Details of the file:", fileName)
        print(X.shape)
    ofile.close()


#run SVM for finger print descriptor file
def svmOnMorganFPFile(svmFile, C = 10, gamma = .01):
    path = str("MorganDataset/" + svmFile)
    try:
        X_train, X_test, y_train, y_test = csr.readCSRFile(path, split=True)
    except:
        X_train, X_test, y_train, y_test = csv.loadDataset(path, split=True)

    dirPath = "svmResults/"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    ofile = open(dirPath  + svmFile, "a")
    ofile.write("Size of the train set: " + str(X_train.shape[0]) + "\n")
    ofile.write("Size of the test set: " + str(X_test.shape[0]) + "\n")
    ofile.close()  # store file size and close, then open again
    ofile = open(dirPath + svmFile, "a")

    # fit svm model
    svm_clf = svmPlus.svmPlusOpt(X_train, y_train, XStar=None, C = C, kernel="rbf",
                             kernelParam = gamma) # standard SVM
    y_predict = svmPlus.predict(X_test, svm_clf)
    correct = np.sum(y_predict == y_test)

    print("Prediction accuracy using SVM for ", svmFile)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))
    predAcc = round(correct / len(y_predict), 3)
    ofile.write("param C = %f, gamma = %f, pred accuracy = %f \n" % (C, gamma, predAcc))
    ofile.close()


# run SVM Plus for finger print descriptor file
def svmPlusOnMorganFPFile(svmFile, svmPlusFile, C=10, gamma=.01,
                          kernelParam=0.0001, kernelParamStar=0.01):
    path = str("MorganDataset/" + svmFile)
    try:
        X_train, X_test, y_train, y_test, indices_train, indices_test = \
            csr.readCSRFile(path, split=True, returnIndices=True)
    except:
        X_train, X_test, y_train, y_test, indices_train, indices_test = \
            csv.loadDataset(path, split = True, returnIndices = True)

    path = str("MorganDataset/" + svmPlusFile)
    try:
        XStar, yStar = csr.readCSRFile(path)
    except:
        XStar, yStar = csv.loadDataset(path)

    XStar_test = XStar[indices_test]
    XStar = XStar[indices_train]
    yStar_test = yStar[indices_test]
    yStar = yStar[indices_train]

    dirPath = "svmPlusResults/"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    ofile = open(dirPath + svmFile, "a")
    ofile.write("Size of the train set: " + str(X_train.shape[0]) + "\n")
    ofile.write("Size of the test set: " + str(X_test.shape[0]) + "\n")
    ofile.close()  # store file size and close, then open again
    ofile = open(dirPath + svmFile, "a")

    # fit svm model
    clf = svmPlus.svmPlusOpt(X_train, y_train, XStar=XStar,
                                C = C, gamma = gamma,
                                kernel="rbf", kernelParam=kernelParam,
                                kernelStar="rbf", kernelStarParam=kernelParamStar)
    y_predict = svmPlus.predict(X_test, clf)
    correct = np.sum(y_predict == y_test)
    print("Prediction accuracy using SVM for ", svmFile)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))
    predAcc = round(correct / len(y_predict), 3)
    ofile.write("param C = %f, gamma = %f, pred accuracy = %f \n" % (C, gamma, predAcc))
    ofile.close()


#enumFileType = enumerate('DESC', '64BITS', '512BITS', '512BITSUNHASHED')
DESC = 0
BITS64 = 1
BITS512 = 2
UNHASHED = 3
#enumDataset = enumerate(BURSI, MMP, CAS)
BURSI = 0
MMP = 1
CAS = 2

#tuned kernel parameters
tunedParam = [[.0001, .1, .1, .01], # BURSI
              [.0001, .1, .001, .01], # MMP
              [.0001, .1, .01, .01]] # CAS

if __name__ == "__main__":
    #readDetailsDescriptorFiles()
    #svmOnMorganFPFile(morgan512BitsUnhashedFiles[2], C = 10, gamma=.01)
    #for fileName in descriptorFiles:
    #    gridSearchWithCV(fileName)
    svmPlusOnMorganFPFile(descriptorFiles[BURSI], morgan512BitsFiles[BURSI],
                          kernelParam = tunedParam[BURSI][DESC],
                          kernelParamStar = tunedParam[BURSI][BITS512])
    #gridSearchSVMPlus(morgan64BitsFiles[0], morgan512BitsFiles[0], morgan512BitsUnhashedFiles[0])
    #svmPlusOnMorganFPFile(descriptorFiles[0], morgan512BitsUnhashedFiles[0])


