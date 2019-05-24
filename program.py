from scipy.optimize import linprog
import numpy as np
import pulp as plp
import math

print("Weaving problem")

# configuration
n = 5 # first dimension of the image
m = 5 # second dimension of the image
resPixel = 5
numWires = 5 

# A = np.zeros((n, m, 3))
A = np.random.uniform(0.0, 1.0, (n,m,3))

for k in range(3):
    print("Original image, color channel %i:" % k)
    print(A[:,:,k])

# wires are not random
w = np.random.uniform(0.0, 1.0, (numWires,3))
for k in range(np.size(w, 0)):
    print("Wire #%i:" % k)
    print(w[k,:])

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

(Ahat, partition) = heuristic(A, w, resPixel)
print("Heuristic solution has error: %.2f" % cost(A, Ahat))

print(A-Ahat)

# prob = plp.LpProblem("A weaving problem", plp.LpMinimize)

# Let's assume for simplicity, the available wires are fixed for now.
# Every pixel in the fabric is divided in resPixel amount of partitions.
# This gives rise to: n*m*resPixel number of variables.
# mapping = {}
# for i in range(n):
    # for j in range(m):
        # for k in range(resPixel):
            # name = "{},{},{}".format(i, j, k)
            # mapping[name] = plp.LpVariable(name, lowBound = 0, upBound = numWires, cat = 'Integer')

# todo: add function we need to optimize

# todo: add constraints

# reading: https://pythonhosted.org/PuLP/CaseStudies/a_sudoku_problem.html
# and doc: https://pythonhosted.org/PuLP/pulp.html

# # pulp hello world
# prob = plp.LpProblem("The Whiskas Problem",plp.LpMinimize)
# x1 = plp.LpVariable("ChickenPercent",0,None,plp.LpInteger)
# x2 = plp.LpVariable("BeefPercent",0)

# # the objective function
# prob += 0.013*x1 + 0.008*x2, "Total Cost of Ingredients per can"

# # some constraints
# prob += x1 + x2 == 100, "PercentagesSum"
# prob += 0.100*x1 + 0.200*x2 >= 8.0, "ProteinRequirement"
# prob += 0.080*x1 + 0.100*x2 >= 6.0, "FatRequirement"
# prob += 0.001*x1 + 0.005*x2 <= 2.0, "FibreRequirement"
# prob += 0.002*x1 + 0.005*x2 <= 0.4, "SaltRequirement"

# prob.writeLP("WhiskasModel.lp")
# prob.solve()
# print("status: ", plp.LpStatus[prob.status])

# for v in prob.variables():
    # print(v.name, " = ", v.varValue)

# print("Total Cost of Ingredients per can = ", plp.value(prob.objective))

# now we have a collection of wire's
# numWires = 2
# w = [0.0, 1.0]

# eta = np.zeros((n,m))

# a hello world example
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
# c = [-1, 4]
# A = [[-3, 1], [1, 2]]
# b = [6, 4]
# x0_bounds = (None, None)
# x1_bounds = (-3, None)
# res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])
# print(res)

# https://stackoverflow.com/questions/26305704/python-mixed-integer-linear-programming


# https://pythonhosted.org/PuLP/CaseStudies/a_blending_problem.html
