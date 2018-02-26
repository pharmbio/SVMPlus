#############################################
# ICP: Inductive Conformal Prediction
#        for Classification using SVM
#############################################
# Import models from scikit learn module:
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from sklearn import svm
import matplotlib as plt
import svmPlusOpt as svmPlus

# compute non-conformity scores
def computeNCWithProb(modelFit,  X_calib, y_calib):

    if(modelFit is None) or (X_calib is None) or (y_calib is None):
        sys.exit("\n NULL model \n")

    predProb = modelFit.predict_proba(X_calib)
    nSamples, nClasses = predProb.shape
    classLabels = [-1, 1]

    MCListConfScores = []  # Moderian Class wise List of conformity scores
    for i in range(0, nClasses):
        clsIndex = np.where(y_calib == classLabels[i])
        classMembers = predProb[clsIndex, i]
        MCListConfScores.append(classMembers)#MCListConfScores[i]+ classMembers.tolist()[0]

    return MCListConfScores


# compute non-conformity scores using decision function
def computeNC(modelFit,  X_calib, y_calib):

    if(modelFit is None) or (X_calib is None) or (y_calib is None):
        sys.exit("\n NULL model \n")

    scores = modelFit.decision_function(X_calib)
    classLabels = [-1, 1]

    MCListConfScores = []  # Moderian Class wise List of conformity scores
    for i in range(0, len(classLabels)):
        clsIndex = np.where(y_calib == classLabels[i])
        classMembers = classLabels[i] * scores[clsIndex]
        MCListConfScores.append(classMembers)#MCListConfScores[i]+ classMembers.tolist()[0]

    return MCListConfScores


# compute non-conformity scores using decision function
def computeNCSVMPlus(clf,  X_calib, y_calib):

    if(clf is None) or (X_calib is None) or (y_calib is None):
        sys.exit("\n NULL model \n")

    scores = svmPlus.decision_function(X_calib, clf)
    classLabels = [-1, 1]
    MCListConfScores = []  # Moderian Class wise List of conformity scores
    for i in range(0, len(classLabels)):
        clsIndex = np.where(y_calib == classLabels[i])
        classMembers = classLabels[i] * scores[clsIndex]
        MCListConfScores.append(classMembers)#MCListConfScores[i]+ classMembers.tolist()[0]

    return MCListConfScores

# compute p-values
def computePValues(MCListConfScores, testConfScores):

    if (MCListConfScores is None) or (testConfScores is None):
        sys.exit("\n NULL model \n")

    nTest = len(testConfScores)
    nClasses = 2
    pValues = np.zeros((nTest,  nClasses))
    classLabels = [-1, 1]

    for k in range(0, nTest):
        for l in range(0, nClasses):
            alpha = classLabels[l] * testConfScores[k]
            classConfScores = MCListConfScores[l]
            pVal = len(classConfScores[np.where(classConfScores < alpha)]) + (np.random.uniform(0, 1, 1) * \
                len(classConfScores[np.where(classConfScores == alpha)]))
            pValues[k, l] = pVal/(len(classConfScores) + 1)

    return(pValues)



def ICPClassification(X, y, X_testData,
                      C = 10, gamma = .01,
                      XStar = None, K = 10, KStar = .01,
                      X_calib = None, y_calib = None):
    if (X is None) or (X_testData is None):
        sys.exit("\n 'trainingSet' and 'testSet' are required as input\n")

    if X_calib is None:
        X_properTrain, X_calib, y_properTrain, y_calib, indices_train, indices_test =\
            train_test_split(X, y, range(len(X)) ,test_size=0.2,
                             stratify=y, random_state=7)
        if XStar is not None:
            XStar_train = XStar[indices_train]

    else:
        X_properTrain = X
        y_properTrain = y
        XStar_train = XStar

    if XStar is not None:
        return SVMPlusICP(X_properTrain, y_properTrain, XStar_train, X_testData,
                          C, gamma, K=K, KStar=KStar,
                          X_calib=X_calib, y_calib=y_calib)
        # commented, this was using the prediction probability
    '''
    model = svm.SVC(gamma=gamma, C=C, probability=True)
    model.fit(X_properTrain , y_properTrain)

    MCListConfScores = computeNC(model, X_calib, y_calib)
    testConfScores = model.predict_proba(X_testData)
    pValues = computePValues(MCListConfScores, testConfScores)
    '''

    # Now, using the score function
    model = svm.SVC(gamma=gamma, C=C)
    model.fit(X_properTrain , y_properTrain)
    MCListConfScores = computeNC(model, X_calib, y_calib)
    #testConfScores = model.predict_proba(X_testData)
    testConfScores = model.decision_function(X_testData)
    pValues = computePValues(MCListConfScores, testConfScores)

    return pValues


def SVMPlusICP(X_properTrain, y_properTrain, XStar_train, X_testData,
               C = 10, gamma = .01,
               K = 10, KStar = .01,
               X_calib=None, y_calib=None):

    clf = svmPlus.svmPlusOpt(X_properTrain, y_properTrain, XStar=XStar_train,
                             C=C, gamma=gamma,
                             kernel="rbf", kernelParam = K,
                             kernelStar="rbf", kernelStarParam = KStar)
    MCListConfScores = computeNCSVMPlus(clf, X_calib, y_calib)
    testConfScores = svmPlus.decision_function(X_testData, clf)
    pValues = computePValues(MCListConfScores, testConfScores)

    return pValues


