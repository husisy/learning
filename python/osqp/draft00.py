import numpy as np
import scipy.sparse

import osqp

P = scipy.sparse.csc_matrix([[4, 1], [1, 2]]) #must be sparse matrix
q = np.array([1, 1])
A = scipy.sparse.csc_matrix([[1, 1], [1, 0], [0, 1]])
l = np.array([1, 0, 0])
u = np.array([1, 0.7, 0.7])
prob = osqp.OSQP()
prob.setup(P, q, A, l, u, alpha=1.0)
res = prob.solve()
res.x
