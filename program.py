import numpy as np
import itertools
from shared import *
import unittest
import cv2
from scipy.cluster.vq import kmeans, whiten, vq

print("Weaving problem")

# configuration
n = 50 # number of pixels
resPixel = 5 # resolution of pixels
numWires = 8 # number of wires that can be used by the machine

# read img
im = cv2.imread("img/lotte")
print("Image shape:", im.shape)

"""
getUniqueColours
img : (n, m, 3) input image
flag : (n, 3) unique colours of the input image
"""
def getUniqueColours(img, max_distance = 1.5):
    flat = 1.0 * np.reshape(img, (np.size(img, 0)*np.size(img,1), 3))

    # calculate clusters of data, to make it easier to work with
    centroids,_ = kmeans(flat, 30)
    # assign each pixel to a cluster
    idx,_ = vq(flat, centroids)

    for k in range(np.size(centroids, 0)):
        subset = flat[idx == k, :] # (N, 3) matrix
        print("center number: %i, elements in cloud: %i" % (k, np.count_nonzero(idx == k)))

        # calculate colours in centroid

    return centroids

uniqueColours = getUniqueColours(im, 5) 
print(uniqueColours)
print("Number of unique colours in image: %i" % len(uniqueColours))

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
# print(uniqueColours[0:100,:])
   
localSearch(uniqueColours/255, w, numWires)

# write img
# im = cv2.imwrite("img/output.png", im)
