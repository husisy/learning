import numpy as np
import cvxopt
# import cvxopt as cvx

import picos

# a list of 8 matrices A
matA = np.array([
    [[1,0,0,0,0], [0,3,0,0,0], [0,0,1,0,0]],
    [[0,0,2,0,0], [0,1,0,0,0], [0,0,0,1,0]],
    [[0,0,0,2,0], [4,0,0,0,0], [0,0,1,0,0]],
    [[1,0,0,0,0], [0,0,2,0,0], [0,0,0,0,4]],
    [[1,0,2,0,0], [0,3,0,1,2], [0,0,1,2,0]],
    [[0,1,1,1,0], [0,3,0,1,0], [0,0,2,2,0]],
    [[1,2,0,0,0], [0,3,3,0,5], [1,0,0,2,0]],
    [[1,0,3,0,1], [0,3,2,0,0], [1,0,0,2,0]],
])
vecC = np.array([1,2,3,4,5])
# c = cvxopt.matrix([1,2,3,4,5])

## multi-response c-optimal design (SOCP)
AA = [picos.Constant(f'A[{i}]', x) for i,x in enumerate(matA)]
s = len(AA)
cc = picos.Constant('c', vecC)
z = [picos.RealVariable(f'z[{i}]', matA.shape[1]) for i in range(s)]
mu = picos.RealVariable('mu', s)
c_primal_SOCP = picos.Problem()
c_primal_SOCP.add_list_of_constraints([abs(z[i]) <= mu[i] for i in range(s)])
c_primal_SOCP.add_constraint(picos.sum([AA[i].T * z[i] for i in range(s)]) == cc)
c_primal_SOCP.set_objective('min', picos.sum(mu) )
c_primal_SOCP.solve(solver='cvxopt')
mu.np / mu.np.sum()

