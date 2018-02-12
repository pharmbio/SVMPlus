# to check how to load image dataset
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import svmOpt as svm
import  svmUtils as utils


digits = load_digits()
print(digits.data.shape)

#plt.gray()
#plt.matshow(digits.images[100])
#plt.show()

print(set(digits.target))
imageZero = digits.images[digits.target == 0]
nZero = len(imageZero)
print(nZero)
imageOne = digits.images[digits.target == 1]
nOne = len(imageOne)
print(nOne)
nSample = 50
X_train = np.vstack((imageZero[:nSample], imageOne[:nSample]))
X_test = np.vstack((imageZero[nSample:], imageOne[nSample:]))
y_train = np.concatenate((np.ones(nSample), -np.ones(nSample)))
y_test =  np.concatenate((np.ones(nZero - nSample), -np.ones(nOne - nSample)))
print(len(y_train))
print(len(y_test))
print(X_train.shape)
print(X_test.shape)

'''
clf = svm.svmOptProb(X_train, y_train, kernel="rbf", C=1, param = .05)
y_predict = utils.predict(X_test, clf)
correct = np.sum(y_predict == y_test)
print("Prediction accuracy of SVM+")
print("%d out of %d predictions correct" % (correct, len(y_predict)))
'''