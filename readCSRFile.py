import numpy as np
import svmPlusOpt as svmPlus
import svmUtils as utils
from numpy import linalg as LA
import mxnet as mx

data_train = mx.io.LibSVMIter(data_libsvm = 'DescriptorDataset/dataset_cas_N6512.csr', data_shape=(10,),
                              label_shape=(100,), batch_size=100)
# The data of the first batch is stored in csr storage type
#batch = data_iter.next()
#csr = batch.data[0]
for batch in data_train:
    X = data_train.getdata()
    print(X.shape)
    label = data_train.getlabel()
    #print(len(label))
