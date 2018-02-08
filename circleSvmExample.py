import numpy as np
import matplotlib.pyplot as plt
import svmOpt as svm
import svmUtils as utils
import svmPlusOpt as svmPlus
from cvxopt import matrix
import math

def genCircularSepData(size):
    clsSize = int(size/2)
    r = np.random.uniform(0,6,clsSize) # Radius
    r1 = [math.sqrt(x) for x in r]
    t1 = 2 * math.pi * np.random.uniform(0,1,clsSize)  # Angle
    data1 = np.zeros((clsSize, 2))
    i=0
    for i in range(clsSize):
        data1[i, 0] = r1[i] * math.cos(t1[i]) + 3
        data1[i, 1] = r1[i] * math.sin(t1[i])
        i = i+1

    r = np.random.uniform(0,4,clsSize) # Radius
    r2 = [math.sqrt(x) for x in r]
    t2 = 2 * math.pi * np.random.uniform(3,4,clsSize)  # Angle
    data2 = np.zeros((clsSize, 2))
    i=0
    for i in range(clsSize):
        data2[i, 0] = (r2[i] * math.cos(t2[i])) + 7
        data2[i, 1] = (r2[i] * math.sin(t2[i]))
        i = i+1

    X = np.vstack((data1, data2))
    y = np.concatenate((np.ones(clsSize), -np.ones(clsSize)))
    return  X, y


red = [[0,1], [1,-1], [4,2], [2,0], [4,3], [2,3]]
blue = [[6,0], [7,-1], [8,1], [9,1], [10,0], [8,-1]]
red = np.asanyarray(red)
blue = np.asanyarray(blue)

'''
plt.plot(red[:, 0], red[:, 1], marker='o', markersize=6, linestyle='', color='r', label='Class1')
plt.plot(blue[:, 0], blue[:, 1], marker='x', markersize=6, linestyle='', color='b', label='Class2')
plt.xlim((-4,10))
plt.ylim((-4,5))
plt.title("Example2", fontsize=10)
plt.axhline(color="black")
plt.axvline(color="black")cd
plt.savefig("circleData.png")
plt.show()
'''

X = np.vstack((red, blue))
y = np.concatenate((-1*np.ones(6), np.ones(6)))

clf1 = svm.svmOptProb(X, y)
#utils.plot_contour(X, y, clf1)

XS = np.zeros(12)#-abs(X[:,0] - 5)
#XS = -np.ones(12) #np.concatenate((np.ones(6), 9*np.ones(6)))
XS[2] = 1
XS[4] = 1
XS[6] = 1

XStar = matrix(XS, tc='d')
clf = svmPlus.svmPlusOpt(X, y, XStar, C=100, gamma=.001)
#utils.plot_contour(X, y, clf)



#XS = np.zeros(12) #np.concatenate((np.ones(6), 9*np.ones(6)))

X, y = genCircularSepData(200)
utils.plot_contour(X, y, clf1)
utils.plot_contour(X, y, clf)