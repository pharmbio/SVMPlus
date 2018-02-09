from cvxopt import matrix, solvers
import numpy as np
from sklearn.datasets import make_blobs
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math

# Linear kernel
def linearKernel(x1, x2, param = None):
    return np.dot(x1, x2)


# Polynomial kernel
def polyKernel(x1, x2, param = 2):
    return (1 + np.dot(x1, x2)) ** param


# Radial basis kernel
def rbfKernel(x1, x2, param = 1.0):
    return np.exp(-(LA.norm(x1 - x2) ** 2) * param)



def funcLine(x, w, b, c=0):
    # given x1, return x2 such that [x1,x2] satisfy the line w.x + b = c
    return (-w[0] * x - b + c) / w[1]


#some global constants for drawing supporting/seprating hyperplanes
a = -1
b = 1


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


# to predict
def predict(X, clf):
    return np.sign(project(clf['w'], clf['b'], X, clf))


# for plotting margin
def plotSvmMargin(X, y, clf):
    sv = clf['sv']
    X1_train = X[y == 1]
    X2_train = X[y == -1]

    plt.plot(X1_train[:, 0], X1_train[:, 1], "ro")
    plt.plot(X2_train[:, 0], X2_train[:, 1], "bo")
    plt.scatter(sv[:, 0], sv[:, 1], s=100, c="g")

    # plot the separating line w.x + b = 0
    a0 = a;
    a1 = funcLine(a0, clf['w'], clf['b'])
    b0 = b;
    b1 = funcLine(b0, clf['w'], clf['b'])
    plt.plot([a0, b0], [a1, b1], "k")

    # plot the supporting lines, w.x + b = 1
    a0 = a;
    a1 = funcLine(a0, clf['w'], clf['b'], 1)
    b0 = b;
    b1 = funcLine(b0, clf['w'], clf['b'], 1)
    plt.plot([a0, b0], [a1, b1], "k--")

    # plot the supporting lines, w.x + b = -1
    a0 = a;
    a1 = funcLine(a0, clf['w'], clf['b'], -1)
    b0 = b;
    b1 = funcLine(b0, clf['w'], clf['b'], -1)
    plt.plot([a0, b0], [a1, b1], "k--")

    plt.axis("tight")
    plt.show()


def plot_contour(X, y, clf, fileName = None):
    sv = clf['sv']
    X1_train = X[y == 1]
    X2_train = X[y == -1]

    plt.plot(X1_train[:, 0], X1_train[:, 1], "ro")
    plt.plot(X2_train[:, 0], X2_train[:, 1], "bo")
    #plt.scatter(sv[:, 0], sv[:, 1], s=100, c="g")

    grid1 = 2
    grid2 = 10

    X1, X2 = np.meshgrid(np.linspace(-grid1, grid2, 50), np.linspace(-grid1, grid2, 50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = project(clf['w'], clf['b'], X, clf).reshape(X1.shape)
    plt.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
    plt.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
    plt.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')
    plt.axis("tight")

    if fileName is not None:
        plt.savefig(fileName)
    plt.show()


def genNonLinSepData():
    mean1 = [-1, 2]
    mean2 = [1, -1]
    mean3 = [4, -4]
    mean4 = [-4, 4]
    cov = [[1.0, 0.8], [0.8, 1.0]]
    X1 = np.random.multivariate_normal(mean1, cov, 50)
    X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 50)
    X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
    y2 = np.ones(len(X2)) * -1
    X = np.vstack((X1,X2))
    y = np.concatenate((y1, y2))
    return X, y

def genCircularSepData(size):
    clsSize = int(size/2)
    r = np.random.uniform(0,1,clsSize) # Radius
    r1 = [math.sqrt(x) for x in r]
    t1 = 2 * math.pi * np.random.uniform(0,1,clsSize)  # Angle
    data1 = np.zeros((clsSize, 2))
    i=0
    for i in range(clsSize):
        data1[i, 0] = r1[i] * math.cos(t1[i])
        data1[i, 1] = r1[i] * math.sin(t1[i])
        i = i+1

    r = np.random.uniform(3,4,clsSize) # Radius
    r2 = [math.sqrt(x) for x in r]
    t2 = 2 * math.pi * np.random.uniform(3,4,clsSize)  # Angle
    data2 = np.zeros((clsSize, 2))
    i=0
    for i in range(clsSize):
        data2[i, 0] = r2[i] * math.cos(t2[i])
        data2[i, 1] = r2[i] * math.sin(t2[i])
        i = i+1

    X = np.vstack((data1, data2))
    y = np.concatenate((np.ones(clsSize), -np.ones(clsSize)))
    return  X, y