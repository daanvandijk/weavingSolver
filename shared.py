import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
import math

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

"""
A : (n, 3) matrix of pixels
w : (n, 3) matrix of available wires
resPixel : the amount of partitions in a pixel
returns : (Ahat, partition)
partition : (n, resPixel)
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

"""
Counts number of specified wire used in specified pixel
partition : (n, resPixel)
iPixel : index of the pixel
iWire : index of the wire
"""
def countWire(partition, iPixel, iWire):
    flag = 0
    for k in range(np.size(partition,1)):
        if (partition[iPixel, k] == iWire):
            flag += 1
    return flag

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

A : (m, n) matrix
b : (m, 1) matrix
x : (n, 1) matrix
"""
def linearProblem(A, b):
    m = np.size(A,0)
    n = np.size(A,1)
    if (m != np.size(b, 0)):
        print("Shape A:", np.shape(A))
        print("Shape b:", np.shape(b))
        raise Exception("Sizes of A and b don't match")
    if (1 != np.size(b, 1)):
        raise Exception("b must be a vector")
    # xi = [s, x]^T 

    c = np.concatenate((np.ones((m,1)), np.zeros((n,1))))
    # print("c: ", c)

    # b_ub = [b, -b]
    b_ub = np.concatenate((b,-b))
    # print("b_ub: ", b_ub)

    # A_ub = [-I, A;
    #         -I, -A]
    A_ub = np.zeros((2*m, m+n))
    A_ub[0:m, 0:m] = -np.eye(m)
    A_ub[m:2*m, 0:m] = -np.eye(m)
    # print("A: ", A)
    A_ub[0:m, m:m+n] = A
    A_ub[m:2*m, m:m+n] = -A
    # print("A_ub: ", A_ub)

    # todo; at bounds to the solution

    res = linprog(c, A_ub=A_ub, b_ub=b_ub)
    if (res['success'] == True):
        return res['x'][n:2*n]
    else:
        print(res)
        raise Exception("Linear problem didn't terminate successfully")

"""
Preperation for the local solution, see pdf
i : index to remove
diff : (J, 3) \hat{a}-a
w : (N, 3) proposal wires
partition : (J, resPixel)
returns (alpha, beta)
"""
def prepareLocalSolution(diff, w, partition, i):
    N = np.size(w, 1)
    if (i < 0 or i >= N):
        raise Exception("Index i is invalid")

    J = np.size(diff, 0)
    resPixel = np.size(partition, 1)
    beta = np.zeros((J, 1))
    alpha = diff
    
    # loop through pixels
    for j in range(J):
        c = countWire(partition, j, i)
        alpha[3*j:3*(j+1),:] += w[i]*(c/resPixel)
        beta[j,0] = (c/resPixel)
    print("beta: ", beta)
    print("alpha: ", alpha)
    
    return (alpha, beta)

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

    error = math.inf  

    (Ahat, partition) = heuristic(A, proposalWires, N)
    error = cost(A, Ahat)
    print("Local search #{} cost: {}".format(1, error))

    # now remove a random wire, and solve the optimization problem
    # when using a L1-norm, it becomes a linear problem
    (alpha, beta) = prepareLocalSolution(Ahat-A, proposalWires, partition, 0)
    sol = linearProblem(alpha, beta)
    print("Solution: ", sol)

    # find closest wire available to sol
 
