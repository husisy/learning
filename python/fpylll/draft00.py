import numpy as np

import fpylll

A = fpylll.IntegerMatrix(50, 50)
A.randomize("ntrulike", bits=50, q=127)
A[0].norm() #3564748886669202.5

M = fpylll.GSO.Mat(A)
M.update_gso()
M.get_mu(1,0) #0.815748944429783

L = fpylll.LLL.Reduction(M)
L()
M.get_mu(1,0) #0.41812865497076024
A[0].norm() #24.06241883103193


# from fpylll import IntegerMatrix, FPLLL
fpylll.FPLLL.set_random_seed(1337)
A = fpylll.IntegerMatrix(9, 10)
A.randomize("intrel", bits=10)
print(A)


A = fpylll.IntegerMatrix.from_matrix([[1,2,3,4],[30,4,4,5],[1,-2,3,4]])
t = (1, 2, 5, 5.1)
v0 = fpylll.CVP.closest_vector(A, t)
v0 #(1, 2, 3, 4)
