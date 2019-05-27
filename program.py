# from scipy.optimize import linprog
import numpy as np
from scipy.spatial import ConvexHull
# import pulp as plp
import math
import itertools

print("Weaving problem")

# configuration
n = 10 # first dimension of the image
resPixel = 5
numWires = 5 

# A = np.zeros((n 3))
A = np.random.uniform(0.0, 1.0, (n,3))

for k in range(3):
    print("Original image, color channel %i:" % k)
    print(A[:,k])

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

def cost(A, Ahat):
    return np.linalg.norm(A-Ahat)

"""
w : (n, 3) matrix of wires
x : 3D vector of wire
returns : index of closest wire
"""
def findClosestWire(w, x):
    best_d = math.inf
    best_i = -1
    for i in range(np.size(w, 0)):
        d = np.linalg.norm(w[i,:]-x)
        if (d < best_d):
            best_i = i
            best_d = d
    return best_i

# a heuristic solution
"""
A : (n, m, 3) matrix of pixels
w : (n, 3) matrix of available wires
resPixel : the amount of partitions in a pixel
returns : (Ahat, partition)
partition : (n, m, resPixel)
"""
def heuristic(A, w, resPixel):
    n = np.size(A,0)
    m = np.size(A,1)
    numWires = np.size(w, 0)
    Ahat = np.zeros((n, m, 3))
    partition = np.zeros((n, m, resPixel))
    alpha = 1.0/resPixel

    for i in range(n):
        for j in range(m):
            solPixel = np.zeros((1,3))

            # find best solution for current pixel
            for k in range(resPixel):
                # find closest wire
                index = findClosestWire(w, A[i,j,:])
                solPixel = solPixel + alpha*w[index, :]
                partition[i,j,k] = index

            Ahat[i,j,:] = solPixel

    return (Ahat, partition)

# (Ahat, partition) = heuristic(A, w, resPixel)
# print("Heuristic solution has error: ",  cost(A, Ahat))

"""
Checks if a point is in a convex hull
https://stackoverflow.com/questions/51771248/checking-if-a-point-is-in-convexhull#51786843
"""
def pointInHull(point, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)

"""
w : (n, 3) matrix of available wires
"""
def localSearch(A, w, N):
    hull = ConvexHull(A)
    numWires = np.size(w,0)
    goodWires = (x for x in w if pointInHull(x, hull))

    # for x in goodWires:
        # print(pointInHull(x, hull))
    
localSearch(A, w, 5)
# print(A-Ahat)
