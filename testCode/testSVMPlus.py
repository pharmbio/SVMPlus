import numpy as np
from sklearn.datasets import make_blobs
import svmOpt as svm
import svmUtils as utils
import svmPlusOpt as svmPlus
from cvxopt import matrix

# test linearly separable case
def testSVM():
    #generate linearly sep data
    X, y = make_blobs(n_samples=100, centers=2, center_box=(-4, 4), random_state=9)
    y[y==0] = -1
    XStar = X  # Using X* = X, please replace with your data
    clf = svmPlus.svmPlusOpt(X, y, XStar, kernel="linear", C=10, gamma = 10)
    utils.plotSvmMargin(X, y, clf=clf)

# test linearly separable but overlapping case
def testSVMSoftMargin():
    #generate linearly sep data
    X, y = make_blobs(n_samples=100, centers=2, center_box=(-4, 4), random_state=7)
    y[y == 0] = -1
    XStar = X  # Using X* = X, please replace with your data
    clf = svmPlus.svmPlusOpt(X, y, XStar, kernel="linear", C=1000, gamma = 10)
    #utils.plotSvmMargin(X, y, clf=clf)
    utils.plot_contour(X, y, clf=clf)

def testSVMPlus():
    #generate linearly sep data
    X, y = make_blobs(n_samples=100, centers=2, n_features=4, center_box=(-4, 3), cluster_std=1.5, random_state=7)
    y[y == 0] = -1
    XStar = X[:,2:4]
    X = X[:, 0:2]

    clfSvm = svm.svmOptProb(X, y)
    utils.plotSvmMargin(X, y, clf=clfSvm)

    clf = svmPlus.svmPlusOpt(X, y, XStar, kernel="linear", C=1, gamma = 1)
    utils.plotSvmMargin(X, y, clf=clf)

def testSVMPlus1():
    cBox = 9
    X, y = make_blobs(n_samples=10, centers=2, n_features=2, center_box=(-cBox, 2),  random_state=8)
    y[y == 0] = -1
    #XS = np.concatenate((4*np.ones(50), np.ones(50)))
    XS = np.zeros(100)
    i=0
    for x in X[:,1]:
        if x > 0:
            XS[i] = 1
        else:
            XS[i] = 0
        i = i+1

    XStar = matrix(XS, tc='d')
    clf1 = svm.svmOptProb(X, y)
    #utils.plot_contour(X, y, clf, "svmFig.png")
    clf = svmPlus.svmPlusOpt(X, y, XStar, C=100, gamma = .01)
    #utils.plot_contour(X, y, clf, "svmPlusFig.png")
    X1, y1 = make_blobs(n_samples=500, centers=2, n_features=2, center_box=(-cBox, 2), random_state=8)
    y1[y1 == 0] = -1
    utils.plot_contour(X1, y1, clf1, "images/svmPredict.png")
    utils.plot_contour(X1, y1, clf, "images/svmPlusPredict.png")


def testSVMPlusCirSepdata():
    X, y = utils.genCircularSepData(100)
    clf = svmPlus.svmPlusOpt(X, y, XStar=X, C=10, kernel="poly", kernelParam=2,
                             kernelStar = "poly", kernelStarParam=2, gamma = 1)
    #clf = svm.svmOptProb(X, y, kernel="rbf", C=1000, param  = .5)
    X_test, y_test = utils.genCircularSepData(500)
    y_predict = svmPlus.predict(X_test, clf)
    correct = np.sum(y_predict == y_test)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))
    utils.plot_contour(X, y, clf)
    utils.plot_contour(X_test, y_test, clf)


if __name__ == "__main__":
    #testSVM()
    #testSVMSoftMargin()
    #testSVMPlus()
    #testSVMPlus1()
    testSVMPlusCirSepdata()