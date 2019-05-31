import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
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
A : (n, 3) matrix of pixels
w : (n, 3) matrix of available wires
resPixel : the amount of partitions in a pixel
returns : (Ahat, partition)
partition : (n, m, resPixel)
"""
def heuristic(A, w, resPixel):
    n = np.size(A,0)
    numWires = np.size(w, 0)
    Ahat = np.zeros((n, 3))
    partition = np.zeros((n, resPixel))
    alpha = 1.0/resPixel

    for i in range(n):
        solPixel = np.zeros((1,3))

        # find best solution for current pixel
        for k in range(resPixel):
            # find closest wire
            index = findClosestWire(w, A[i,:])
            solPixel = solPixel + alpha*w[index, :]
            partition[i,k] = index

        Ahat[i,:] = solPixel

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
Solves the following LP-problem
min ||Ax-b||_1 s.t. x in R^n

A : (n, n) matrix
b : (n, 1) matrix
x : (n, 1) matrix
"""
def linearProblem(A, b):
    if (np.size(A,0) != np.size(A,1)):
        raise Exception("Input matrix must be square")
    n = np.size(A,0)
    # xi = [s, x]^T 

    c = np.concatenate((np.ones((n,1)), np.zeros((n,1))))
    # print("c: ", c)

    # b_ub = [b, -b]
    b_ub = np.concatenate((b,-b))
    # print("b_ub: ", b_ub)

    # A_ub = [-I, A;
    #         -I, -A]
    A_ub = np.zeros((2*n, 2*n))
    A_ub[0:n, 0:n] = -np.eye(n)
    A_ub[n:2*n, 0:n] = -np.eye(n)
    # print(A)
    A_ub[0:n, n:2*n] = A
    A_ub[n:2*n, n:2*n] = -A
    # print("A_ub: ", A_ub)

    # todo; at bounds to the solution

    res = linprog(c, A_ub=A_ub, b_ub=b_ub)
    if (res['success'] == True):
        return res['x'][n:2*n]
    else:
        print(res)
        raise Exception("Linear problem didn't terminate successfully")
"""
A : (J, 3) matrix of the target image we want to approximate
w : (I, 3) matrix of available wires
N : number of wires that can be used by the machine
"""
def localSearch(A, w, N):
    hull = ConvexHull(A)
    numWires = np.size(w,0)

    # find the subset wires that are outside the convex hull
    # todo: add boundary points maybe?
    goodWires = []
    for k in range(numWires):
        if (pointInHull(w[k,:], hull)):
            goodWires.append(w[k,:])

    # now pick N random points
    # todo: make this random
    proposalWires = np.zeros((N, 3))
    for k in range(N):
        proposalWires[k,:] = goodWires[k]

    # print(proposalWires)

    error = math.inf  

    (Ahat, partition) = heuristic(A, proposalWires, N)
    error = cost(A, Ahat)
    print("Local search #{} cost: {}".format(1, error))

    # now remove a random wire, and solve the optimization problem
    # quadratic programming: https://cvxopt.org/userguide/coneprog.html#quadratic-programming
    # when using a L1-norm, it becomes a linear problem
    iWire = 0 # remove the zeroth wire
    J = np.size(A, 0)
    Ahat = np.zeros((3*J, 3))
    for k in range(J):
        Ahat[3*k:3*(k+1),:] = np.eye(3)
    print("A: ", Ahat)
    alpha = np.zeros((J*3,1))
    print("b: ", alpha)
    # sol = linearProblem((1/N)*np.ones((J,3)), b)

    # find closest wire available to sol
    
localSearch(A, w, 5)
# print(A-Ahat)
