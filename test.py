import unittest
from shared import *

class testHeuristic(unittest.TestCase):
    def test(self):
        resPixel = 3
        # random input image
        A = np.random.uniform(0.0, 1.0, (10,3))
        # random wires
        w = np.random.normal(0.0, 1.0, (2*resPixel,3))

        (Ahat, partition) = heuristic(A, w, resPixel)
        self.assertTrue(np.size(partition, 1) == resPixel)

class testLinearProblem(unittest.TestCase):
    def testSymmetric(self):
        A = np.eye(2)
        b = np.ones((2,1))
        res = linearProblem(A,b)
        self.assertTrue(np.size(res, 0) == 2)

    def testAsymmetric(self):
        A = np.zeros((4,2))
        b = np.ones((4,1))
        A[0,:] = [0, 1]
        A[1,:] = [0.5, 0.5]
        A[2,:] = [0.75, 0.6]
        A[3,:] = [0.6, 0.3]
        res = linearProblem(A,b)
        self.assertTrue(np.size(res, 0) == 2)


unittest.main()
