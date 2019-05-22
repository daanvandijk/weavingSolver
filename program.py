from scipy.optimize import linprog
import numpy as np

print("Weaving problem")

n = 5
m = 5
R = np.zeros((n, m))
for i in range(n):
    for j in range(m):
        R[i,j] = np.random.uniform(0.0, 1.0)

print("Original R matrix:")
print(R)

# now we have a collection of wire's
numWires = 2
w = [0.0, 1.0]

eta = np.zeros((n,m))

# a hello world example
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
c = [-1, 4]
A = [[-3, 1], [1, 2]]
b = [6, 4]
x0_bounds = (None, None)
x1_bounds = (-3, None)
res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])
print(res)

# https://stackoverflow.com/questions/26305704/python-mixed-integer-linear-programming


# https://pythonhosted.org/PuLP/CaseStudies/a_blending_problem.html
