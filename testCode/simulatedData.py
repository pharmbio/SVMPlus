import numpy as np
import svmPlusOpt as svmPlus
import svmUtils as utils
from numpy import linalg as LA



mean1 = np.zeros(2)
mean2 = np.ones(2)

cov2  = cov1 = .5 * np.eye(2,2)

#cov2 = .15 * np.eye(2,2)
n = 10
X1 = np.random.multivariate_normal(mean1, cov = cov1, size = n)
X2 = np.random.multivariate_normal(mean2, cov = cov2, size = n)

X = np.vstack((X1, X2))
y = np.concatenate((np.ones(n), -np.ones(n)))

clfX = svmPlus.svmPlusOpt(X, y, kernel="rbf", C=100, kernelParam = .1)
#utils.plot_contour(X, y, clf=clfX)

X1Star = np.zeros((n,2))
X2Star = np.zeros((n,2))
i=0
for x in X1:
    X1Star[i, 0] = LA.norm(x - mean1)
    X1Star[i, 1] = LA.norm(x - mean2)
    i = i+1


i=0
for x in X2:
    X2Star[i, 0] = LA.norm(x - mean2)
    X2Star[i, 1] = LA.norm(x - mean1)
    i = i+1

XStar = np.vstack((X1Star, X2Star))

clfXStar = svmPlus.svmPlusOpt(XStar, y, kernel="rbf", C=10, kernelParam  = 1)
#utils.plot_contour(XStar, y, clf=clfXStar)

clf = svmPlus.svmPlusOpt(X, y, XStar, kernel="rbf", kernelParam = .1,
                         kernelStar = "rbf", kernelStarParam = .1,
                         C=.01, gamma = .0001)
#utils.plot_contour(X, y, clf=clf)

ntest = 50
#Predict
X1 = np.random.multivariate_normal(mean1, cov = cov1, size = ntest)
X2 = np.random.multivariate_normal(mean2, cov = cov2, size = ntest)
X_test = np.vstack((X1, X2))
y_test = np.concatenate((np.ones(ntest), -np.ones(ntest)))

y_predict = svmPlus.predict(X_test, clfX)
correct = np.sum(y_predict == y_test)
print("Prediction accuracy of SVM on X")
print("%d out of %d predictions correct" % (correct, len(y_predict)))

y_predict = svmPlus.predict(X_test, clfXStar)
correct = np.sum(y_predict == y_test)
print("Prediction accuracy of SVM on XStar")
print("%d out of %d predictions correct" % (correct, len(y_predict)))

y_predict = svmPlus.predict(X_test, clf)
correct = np.sum(y_predict == y_test)
print("Prediction accuracy of SVM+")
print("%d out of %d predictions correct" % (correct, len(y_predict)))
