# Experiment with image dataset for image classification
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import svmOpt
import svmUtils as utils
from sklearn import svm

digits = load_digits()
print(digits.data.__class__)
print(digits.data.shape)

# to plot image
#plt.gray()
#plt.matshow(digits.images[100])
#plt.show()

#print(set(digits.target))
dataClass1 = digits.data[digits.target == 4]
nClass1 = len(dataClass1)

dataClass2 = digits.data[digits.target == 9]
nClass2 = len(dataClass2)
#print(nOne)
nSample = 50
X_train = np.vstack((dataClass1[:nSample], dataClass2[:nSample]))
X_test = np.vstack((dataClass1[nSample:], dataClass2[nSample:]))
y_train = np.concatenate((np.ones(nSample), -np.ones(nSample)))
y_test =  np.concatenate((np.ones(nClass1 - nSample), -np.ones(nClass2 - nSample)))


# train and predict using SVM implemented in sklearn
clf = svm.SVC(gamma = .0001, C = 100.)
clf.fit(X_train, y_train)
print(len(clf.support_), "support vectors")

y_predict = clf.predict(X_test)
print(sum(y_predict == y_test), "prediction accuracy using sklearn.svm")

# train and predict using SVM implemented
clf = svmOpt.svmOptProb(X_train, y_train, kernel="rbf", C=100, param = .0001)
y_predict = utils.predict(X_test, clf)
correct = np.sum(y_predict == y_test)
print("Prediction accuracy of SVM")
print("%d out of %d predictions correct" % (correct, len(y_predict)))
