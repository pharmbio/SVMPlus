# SVM Optimization method implementation by Niharika Gauraha

from cvxopt import matrix, solvers
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import svmUtils as utils

# svm Optimization Problem
# X: design matrix
# y:  labels {+1, -1}
# K: kernel
# C: tuning parameter for slack variables
def svmOptProb(X, y, kernel="linear", C=None, param = None):
    nSamples, nFeatures = X.shape

    if kernel == "linear":
        kernelMethod = utils.linearKernel
    elif kernel == "poly":
        kernelMethod = utils.polyKernel
    else:
        kernelMethod = utils.rbfKernel

    # compute the matrix K (nSamples X nSamples) using kernel function
    K = np.zeros((nSamples, nSamples))
    for i in range(nSamples):
        for j in range(nSamples):
            K[i, j] = kernelMethod(X[i,:], X[j,:], param)

    P = matrix(np.outer(y, y) * K)
    q = matrix(np.ones(nSamples)* -1)
    A = matrix(y, (1, nSamples), tc="d")
    b = matrix(0.0)

    if C is None:
        G = matrix(np.diag(np.ones(nSamples) * -1))
        h = matrix(np.zeros(nSamples))
    else:
        tmp1 = np.diag(np.ones(nSamples) * -1)
        tmp2 = np.identity(nSamples)
        G = matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(nSamples)
        tmp2 = np.ones(nSamples) * C
        h = matrix(np.hstack((tmp1, tmp2)))


    # solve the QP problem
    sol = solvers.qp(P, q, G, h, A, b)

    # Lagrange multipliers
    alpha = np.ravel(sol['x'])

    #alpha = np.array(sol['x'])
    # Support vectors have non zero lagrange multipliers
    sv = alpha > 1e-3  # tolerance
    ind = np.arange(len(alpha))[sv]
    alpha = alpha[sv]
    print("%d support vectors out of %d points" % (len(alpha), nSamples))
    sv_x = X[sv]
    sv_y = y[sv]

    bias = 0

    for n in range(len(alpha)):
        bias += sv_y[n] - np.sum(alpha * sv_y * K[ind[n], sv])

    bias /= len(alpha)

    # Weight vector
    if kernel=="linear":
        w = np.zeros(nFeatures)
        for n in range(len(alpha)):
            w += alpha[n] * sv_y[n] * sv_x[n]
    else:
        w = None

    clf = {}
    clf['sv_x'] = sv_x
    clf['sv_y'] = sv_y
    clf['alpha'] = alpha
    clf['w'] = w
    clf['b'] = bias
    clf['kernel'] = kernelMethod
    clf['kernelParam'] = param

    return clf



def funcLine(x, w, b, c=0):
    # given x1, return x2 such that [x1,x2] satisfy the line w.x + b = c
    return (-w[0] * x - b + c) / w[1]

# for plotting margin
def plotSvmMargin(X, y, clf):
    sv = clf['sv_x']
    X1_train = X[y == 1]
    X2_train = X[y == -1]

    plt.plot(X1_train[:, 0], X1_train[:, 1], "ro")
    plt.plot(X2_train[:, 0], X2_train[:, 1], "bo")
    plt.scatter(sv[:, 0], sv[:, 1], s=100, c="g")

    # plot the separating line w.x + b = 0
    a0 = -8;
    a1 = funcLine(a0, clf['w'], clf['b'])
    b0 = 4;
    b1 = funcLine(b0, clf['w'], clf['b'])
    plt.plot([a0, b0], [a1, b1], "k")

    # plot the supporting lines, w.x + b = 1
    a0 = -8;
    a1 = funcLine(a0, clf['w'], clf['b'], 1)
    b0 = 4;
    b1 = funcLine(b0, clf['w'], clf['b'], 1)
    plt.plot([a0, b0], [a1, b1], "k--")

    # plot the supporting lines, w.x + b = -1
    a0 = -8;
    a1 = funcLine(a0, clf['w'], clf['b'], -1)
    b0 = 4;
    b1 = funcLine(b0, clf['w'], clf['b'], -1)
    plt.plot([a0, b0], [a1, b1], "k--")

    plt.axis("tight")
    plt.show()

# test linearly separable case
def testSVM():
    #generate linearly sep data
    X, y = make_blobs(n_samples=100, centers=2, center_box=(-4, 4), random_state=9)
    y[y==0] = -1
    clf = svmOptProb(X, y)
    #plotSvmMargin(X, y, clf)
    utils.plot_contour(X, y, clf)

# test linearly separable but overlapping case
def testSVMSoftMargin():
    #generate linearly sep data
    X, y = make_blobs(n_samples=100, centers=2, center_box=(-4, 3), random_state=9)
    y[y==0] = -1
    clf = svmOptProb(X, y, C=1)
    utils.plot_contour(X, y, clf=clf)
    #plotSvmMargin(X, y, clf)


def testSVMNonLinSep():
    X, y = utils.genNonLinSepData()
    y[y==0] = -1
    clf = svmOptProb(X, y, kernel="rbf", C=1)
    utils.plot_contour(X, y, clf)


def testSVMCirSepdata():
    X, y = utils.genCircularSepData(100)
    clf = svmOptProb(X, y, kernel="poly", C=1, param = 2)
    #clf = svmOptProb(X, y, kernel="rbf", C=1000, param  = .5)
    utils.plot_contour(X, y, clf)


if __name__ == "__main__":
    #testSVMCirSepdata()
    #testSVMSoftMargin()
    testSVM()