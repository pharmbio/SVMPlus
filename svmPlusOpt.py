# SVM Optimization Problem implementation by me: Niharika Gauraha

from cvxopt import matrix, solvers
import numpy as np
from sklearn.datasets import make_blobs
import svmOpt as svm
import svmUtils as utils
from numpy.matlib import repmat
from scipy.spatial.distance import pdist, squareform
import scipy
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import sys
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel


# svm Optimization Problem
# X: design matrix
# y:  labels {+1, -1}
# K: kernel
# C: tuning parameter for slack variables
def svmPlusOpt(X, y, XStar=None, C=10, kernel="linear", kernelParam = None,
               kernelStar = "linear", kernelStarParam = None, gamma = 10):
    nSamples, nFeatures = X.shape

    # if XStart is not passed then call standard SVM
    if XStar is None:
        clf = svm.svmOptProb(X, y, C=C, kernel=kernel, param=kernelParam)
        return  clf

    if kernel == "linear":
        kernelMethod = utils.linearKernel
    elif kernel == "poly":
        kernelMethod = utils.polyKernel
    else:
        kernelMethod = utils.rbKernel

    if kernelStar == "linear":
        kernelStarMethod = utils.linearKernel
    elif kernelStar == "poly":
        kernelStarMethod = utils.polyKernel
    else:
        kernelStarMethod = utils.rbKernel

    # compute the matrix K (nSamples X nSamples) using kernel function
    K = np.zeros((nSamples, nSamples))
    KStar = np.zeros((nSamples, nSamples))
    for i in range(nSamples):
        for j in range(nSamples):
            K[i, j] = kernelMethod(X[i,:], X[j,:], kernelParam)
            #KStar[i, j] =  np.dot(XStar[i, :], XStar[j, :])# Assuming linear kernel as of now.
            KStar[i, j] = kernelStarMethod(XStar[i, :], XStar[j, :], kernelStarParam)

    KSvm = matrix(np.outer(y, y) * K)
    P1 = np.concatenate((KSvm + KStar / float(gamma), KStar / float(gamma)), axis=1)
    P2 = np.concatenate((KStar / float(gamma),KStar / float(gamma)), axis=1)
    P = np.concatenate((P1, P2), axis=0)
    A = np.concatenate((np.ones((1, 2 * nSamples)), np.concatenate((np.transpose(matrix(y)), np.zeros((1, nSamples))), axis=1)),
                        axis=0)
    b = np.array([[nSamples * C], [0]])
    G = -np.eye(2 * nSamples)
    h = np.zeros((2 * nSamples, 1))

    Q = repmat(sum(KStar + np.transpose(KStar)), 1, 2) * (
            -C / float(2 * gamma)) - \
        np.concatenate((np.ones((1, nSamples)), np.zeros((1, nSamples))), axis=1)
    q = np.transpose(Q)

    print(P.shape)
    print(Q.shape)
    print(A.shape)
    print(G.shape)
    print(h.shape)
    print(b.shape)
    sol = solvers.qp(matrix(P, tc='d'), matrix(q, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'), matrix(A, tc='d'),
                     matrix(b, tc='d'))
    # Lagrange multipliers
    alpha = np.ravel(sol['x'][0:nSamples])

    # Support vectors have non zero lagrange multipliers
    sv = alpha > 2e-3
    ind = np.arange(len(alpha))[sv]
    alpha = alpha[sv]
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
    clf['K'] = K
    clf['sv_x'] = sv_x
    clf['sv_y'] = sv_y
    clf['sv'] = sv_x
    clf['alpha'] = alpha
    clf['w'] = w
    clf['b'] = bias
    clf['kernel'] = kernelMethod

    return clf


def project(w, b, X, clf):
    if w is not None:
        return np.dot(X, w) + b
    else:
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(clf['alpha'], clf['sv_y'], clf['sv']):
                s += a * sv_y * clf['kernel'](X[i], sv)
            y_predict[i] = s
        return y_predict + b


def predict(X, clf):
    return np.sign(project(clf['w'], clf['b'], X, clf))

