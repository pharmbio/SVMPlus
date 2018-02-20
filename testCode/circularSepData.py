import numpy as np
import matplotlib.pyplot as plt
import math

# To generate 100 points uniformly distributed in the disk of radius r, first generate a radius r as the square root of
# a uniform random variable, generate an angle t uniformly in (0, r), and put the point at (r cos( t ), r sin( t )).

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


if __name__ == "__main__":
    X, y = genCircularSepData(100)
    plt.plot(X[:50, 0], X[:50, 1], "ro")
    plt.plot(X[51:, 0], X[51:, 1], "bo")
    plt.show()