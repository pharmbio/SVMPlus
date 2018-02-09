from sklearn.model_selection import train_test_split
from numpy.random import RandomState
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=50, centers=2, n_features=2, center_box=(-5, 5),  random_state=8)
X1, y1 = make_blobs(n_samples=50, centers=2, n_features=2, center_box=(-5, 5),  random_state=8)

indices = range(50)

random_state = RandomState()
X_train, X_test, y_train, y_test, indices_train,indices_test = \
    train_test_split(X, y, indices, test_size=0.3, random_state = random_state)
X_train1 = X[indices_train]
X_test1 = X[indices_test]
y_train1 = y[indices_train]
y_test1 = y[indices_test]

if (y_train1 == y_train).all():
    print("Rand state works")
else:
    print("It does not work")