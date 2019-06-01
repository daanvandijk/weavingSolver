import numpy as np
import itertools
from shared import *
import unittest

print("Weaving problem")

# configuration
n = 10 # number of pixels
resPixel = 5
numWires = 5 

# A = np.zeros((n 3))
A = np.random.uniform(0.0, 1.0, (n,3))

# for k in range(3):
    # print("Original image, color channel %i:" % k)
    # print(A[:,k])

def generateWires(k):
    X = np.linspace(0.0, 1.0, k)
    w = np.zeros((k*k*k,3))
    index = 0
    for i,j,k in itertools.product(range(k), range(k), range(k)):
        w[index,0] = X[i]
        w[index,1] = X[j]
        w[index,2] = X[k]
        index += 1
    return w

w = generateWires(10)
print("Number of wires: ", np.size(w, 0))
   
localSearch(A, w, 5)
# print(A-Ahat)
