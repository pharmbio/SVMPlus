import numpy as np
import svmPlusOpt as svmPlus
import svmOpt
import svmUtils as utils
from numpy import linalg as LA
import mxnet as mx
import sklearn.svm as svm

# CAS dataset, shape:31581
# MMP datase, shape : 33237
# BURSI dataset, shape : 22890
def loadSigData():
    # number of features: 31581
    #save it for CAS dataset
    #6400
    '''
    data_train = mx.io.LibSVMIter(data_libsvm = 'DescriptorDataset/dataset_cas_N6512.csr', data_shape=(31581,),
                                  label_shape=(1,), batch_size=100, shuffle=False, last_batch_handle="discard")
    '''
    # rows:5550, for MMP dataset
    '''
    data_train = mx.io.LibSVMIter(data_libsvm='DescriptorDataset/dataset_mmp.csr', data_shape=(33237,),
                                  label_shape=(1,), batch_size=100, shuffle=False, last_batch_handle="discard")
    '''
    data_train = mx.io.LibSVMIter(data_libsvm = 'DescriptorDataset/dataset_bursi.csr', data_shape=(22890,),
                                  num_parts = 100,
                                  label_shape=(1,), batch_size=10, shuffle=False, last_batch_handle="discard")
    '''

    data_train = mx.io.LibSVMIter(data_libsvm = 'DescriptorDataset/dataset_temp.csr', data_shape=(22890,),
                                      label_shape=(1,), batch_size=5, shuffle=False, round_batch=True)
    '''
    # The data of the first batch is stored in csr storage type
    #batch = data_iter.next()
    #csr = batch.data[0]
    X_train = None
    y_train = None
    for i in range(100):
        for batch in data_train:
            #X = data_train.getdata()
            X = batch.data[0]
            #label = data_train.getlabel()
            label = batch.label[0]
            if X_train is None:
                X_train = X.asnumpy()
                y_train = label.asnumpy()
            else:
                X_train = np.vstack((X_train, X.asnumpy()))
                y_train = np.concatenate((y_train, label.asnumpy()))
        if(i != 99):
            data_train.reset()
            data_train.next()

    y_train = y_train.astype(int)
    y_train[y_train==0] = -1

    return X_train, y_train


def getSigData():
    # number of features: 31580+1(label)
    data_train = mx.io.LibSVMIter(data_libsvm = 'DescriptorDataset/dataset_cas_N6512.csr', data_shape=(31581,),
                                  label_shape=(1,), batch_size=1000)
    # The data of the first batch is stored in csr storage type
    #batch = data_iter.next()
    #csr = batch.data[0]
    #for batch in data_train:
    X = data_train.getdata()
    #print(X.shape)
    X_train = X.asnumpy()
    label = data_train.getlabel()
    y = label.asnumpy()
    y_train = y.astype(int)
    y_train[y_train==0] = -1

    batch = data_train.next() # ignore once
    batch = data_train.next()
    X = batch.data[0]
    #print(X.shape)
    X_test = X.asnumpy()
    label = batch.label[0]
    y = label.asnumpy()
    y_test = y.astype(int)
    y_test[y_train==0] = -1

    return X_train, X_test, y_train, y_test



def testSVMSigDesc():
    X_train, X_test, y_train, y_test =getSigData()
    # train and predict using SVM implemented in sklearn
    clf = svm.SVC(gamma=.01, C=100.)
    clf.fit(X_train, y_train)
    print(len(clf.support_), "support vectors")

    y_predict = clf.predict(X_test)
    print(sum(y_predict == y_test), "prediction accuracy using sklearn.svm")

    # train and predict using SVM implemented
    clf = svmOpt.svmOptProb(X_train, y_train, kernel="rbf", C=100, param=.01)
    y_predict = utils.predict(X_test, clf)
    correct = np.sum(y_predict == y_test)
    print("Prediction accuracy of SVM")
    print("%d out of %d predictions correct" % (correct, len(y_predict)))


#testSVMSigDesc()

X, y = loadSigData()
print(X.shape)
print(X[0, :20])
print(X[2, 0:40])
print(X[len(X)-3, 0:40])
print(X[len(X)-2, 0:40])
print(X[4336, 0:100])
print(len(y))
