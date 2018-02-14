from mnist import MNIST
import numpy as np
import svmOpt
import svmUtils as utils
from sklearn import svm
import csv
import matplotlib.pyplot as plt
import scipy.ndimage
import svmPlusOpt as svmPlus


def images2numpy(images):
    images_np = np.array(images)
    #images_np = ((images_np / 255) > 0.5).astype('int')  # bool to 0/1
    return images_np


def resizeImage(image):
    print(image.shape)
    originalImage = image.reshape(28,28)
    # to plot image original image
    '''
    #plt.gray()
    #plt.matshow(originalImage )
    #plt.show()
    '''
    resizedImage = scipy.ndimage.zoom(originalImage, 1/(2.8), order=0)
    # to plot resized image
    '''
    #plt.gray()
    #plt.matshow(resizedImage)
    #plt.show()
    #print(resizedImage.shape) #to confirm it is resized to 10by10
    '''
    return resizedImage.reshape(100, )

# Reads original MNIST data and stores images only for digits 5 and 8
def preProcessMNISTData():
    mndata = MNIST('MNISTData')
    train_data,  y_train = mndata.load_training()
    #test_data,  y_test = mndata.load_testing()

    train_data = images2numpy(train_data)
    train_label = np.array(y_train)

    dataClass1 = train_data[train_label == 5]
    nClass1 = len(dataClass1)
    dataClass2 = train_data[train_label == 8]
    nClass2 = len(dataClass2)

    X = np.vstack((dataClass1, dataClass2))
    print(X.shape)
    y = np.concatenate((np.ones(nClass1), -np.ones(nClass2)))
    np.savetxt("data/mnistData.csv", X, delimiter=",")
    np.savetxt("data/mnistLabel.csv", y, delimiter=",")


# to test if the above function correctly stored the digits 5 and 8
# and run svm on the data
def testMNIST():
    ifile = open("data/mnistData.csv")
    reader = csv.reader(ifile)
    a = []
    for row in reader:
        newRow = [float(val) if val else 0 for val in row]
        a.append(newRow)
    ifile.close()
    X = np.array([ x for x in a]).astype(float)
    print(X.shape)

    ifile = open("data/mnistLabel.csv")
    reader = csv.reader(ifile)
    a = []
    for row in reader:
        a.append(row)
    ifile.close()
    y = np.array(a).astype(float).reshape(1, 11272)
    y = np.array([ x for x in y]).astype(int)
    y = y[0]
    print(y)

    print(sum(y == 1))
    dataClass1 = X[y == 1]
    nClass1 = len(dataClass1)
    print(nClass1)
    dataClass2 = X[y == -1]
    nClass2 = len(dataClass2)
    print(nClass2)

    #train_data, y_train = mndata.load_training()
    nSample = 50
    X_train = np.vstack((dataClass1[:nSample], dataClass2[:nSample]))
    X_test = np.vstack((dataClass1[nSample:], dataClass2[nSample:]))
    y_train = np.concatenate((np.ones(nSample), -np.ones(nSample)))
    y_test =  np.concatenate((np.ones(nClass1 - nSample), -np.ones(nClass2 - nSample)))
    print(X_train.shape)
    print(X_test.shape)
    print(len(y_test))
    
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


# resize the original 28by28 images into 10by10 images
# and save it in a file
def resizeMNISTData():
    ifile = open("data/mnistData.csv")
    reader = csv.reader(ifile)
    a = []
    for row in reader:
        newRow = [float(val) if val else 0 for val in row]
        a.append(newRow)
    ifile.close()
    X = np.array([ x for x in a]).astype(float)
    print(X.shape)

    #resizeImage(X[0])
    #for i in range(len(X)):
    X_resized = np.array([resizeImage(x) for x in X])
    np.savetxt("data/mnistResizedData.csv", X_resized, delimiter=",")


def testResizedMNISTDataset():
    ifile = open("data/mnistResizedData.csv")
    reader = csv.reader(ifile)
    a = []
    for row in reader:
        newRow = [float(val) if val else 0 for val in row]
        a.append(newRow)
    ifile.close()
    X = np.array([x for x in a]).astype(float)

    ifile = open("data/mnistLabel.csv")
    reader = csv.reader(ifile)
    a = []
    for row in reader:
        a.append(row)
    ifile.close()
    y = np.array(a).astype(float).reshape(1, 11272)
    y = np.array([x for x in y]).astype(int)
    y = y[0]

    dataClass1 = X[y == 1]
    nClass1 = len(dataClass1)
    dataClass2 = X[y == -1]
    nClass2 = len(dataClass2)

    # train_data, y_train = mndata.load_training()
    nSample = 50
    X_train = np.vstack((dataClass1[:nSample], dataClass2[:nSample]))
    X_test = np.vstack((dataClass1[nSample:], dataClass2[nSample:]))
    y_train = np.concatenate((np.ones(nSample), -np.ones(nSample)))
    y_test = np.concatenate((np.ones(nClass1 - nSample), -np.ones(nClass2 - nSample)))
    print(X_train.shape)
    print(X_test.shape)
    print(len(y_test))

    # train and predict using SVM implemented in sklearn
    clf = svm.SVC(gamma=.0001, C=100.)
    clf.fit(X_train, y_train)
    print(len(clf.support_), "support vectors")

    y_predict = clf.predict(X_test)
    print(sum(y_predict == y_test), "prediction accuracy using sklearn.svm")

    # train and predict using SVM implemented
    clf = svmOpt.svmOptProb(X_train, y_train, kernel="rbf", C=100, param=.0001)
    y_predict = utils.predict(X_test, clf)
    correct = np.sum(y_predict == y_test)
    print("Prediction accuracy of SVM")
    print("%d out of %d predictions correct" % (correct, len(y_predict)))


# run SVM on X, SVM on XStar and SVM+ on X (using XStar as prib-info)
def testSVMPlusMNISTDataset():
    ifile = open("data/mnistResizedData.csv")
    reader = csv.reader(ifile)
    a = []
    for row in reader:
        newRow = [float(val) for val in row]
        a.append(newRow)
    ifile.close()
    X = np.array([x for x in a]).astype(float)

    ifile = open("data/mnistLabel.csv")
    reader = csv.reader(ifile)
    a = []
    for row in reader:
        a.append(row)
    ifile.close()
    y = np.array(a).astype(float).reshape(1, 11272)
    y = np.array([x for x in y]).astype(int)
    y = y[0]

    dataClass1 = X[y == 1]
    #nClass1 = len(dataClass1)
    dataClass2 = X[y == -1]
    #nClass2 = len(dataClass2)

    # 50+50 for training
    nSample = 50
    testSize = 933
    X_train = np.vstack((dataClass1[:nSample], dataClass2[:nSample]))
    X_test = np.vstack((dataClass1[nSample:(nSample+testSize)], dataClass2[nSample:(nSample+testSize)]))
    y_train = np.concatenate((np.ones(nSample), -np.ones(nSample)))
    y_test = np.concatenate((np.ones(testSize), -np.ones(testSize)))

    # train and predict using SVM implemented in sklearn
    clf = svm.SVC(gamma=.000001, C=100.)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print("Prediction accuracy of SVM on X")
    print("number of test error = %d, test size = %d" % (len(y_predict) - correct, len(y_predict)))

    #Read XStar
    ifile = open("data/mnistData.csv")
    reader = csv.reader(ifile)
    a = []
    for row in reader:
        newRow = [float(val) for val in row]
        a.append(newRow)
    ifile.close()
    XStar = np.array([x for x in a]).astype(float)
    dataClass1 = XStar[y == 1]
    dataClass2 = XStar[y == -1]
    XStar_train = np.vstack((dataClass1[:nSample], dataClass2[:nSample]))
    XStar_test = np.vstack((dataClass1[nSample:(nSample+testSize)], dataClass2[nSample:(nSample+testSize)]))

    # train and predict using SVM on XStar
    clf = svmPlus.svmPlusOpt(XStar_train, y_train, XStar=None, C=100, kernel="rbf",
                             kernelParam=0.000001)
    y_predict = svmPlus.predict(XStar_test, clf)
    correct = np.sum(y_predict == y_test)
    print("Prediction accuracy of SVM on XStar")
    print("number of test error = %d, test size = %d" % (len(y_predict) - correct, len(y_predict)))

    # compute prediction accuracy using SVM+ on X, and Star as priv-info
    clf = svmPlus.svmPlusOpt(X_train, y_train, XStar=XStar_train, C=100, kernel="rbf",
                             kernelParam=0.000001, kernelStar="rbf", kernelStarParam=0.000001,
                             gamma=0.001)
    y_predict = svmPlus.predict(X_test, clf)
    correct = np.sum(y_predict == y_test)
    print("Prediction accuracy of SVM+ on X using XStar as priv-info")
    print("number of test error = %d, test size = %d" %(len(y_predict) - correct, len(y_predict)))


testSVMPlusMNISTDataset()