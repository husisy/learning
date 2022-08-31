import numpy as np
import cvxopt
# from cvxopt import matrix

# A = cvxopt.matrix([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2,3))
A = cvxopt.matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) #column major
A.size #(3,2)
npA = np.array(A)



## linear programming
# A*x=b, minimize(c*x)
A = cvxopt.matrix([ [-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, -1.0, -2.0] ])
b = cvxopt.matrix([ 1.0, -2.0, 0.0, 4.0 ])
c = cvxopt.matrix([ 2.0, 1.0 ])
sol = cvxopt.solvers.lp(c, A, b)
sol['x']


## quadratic program
# 0.5*x*Q*x + p*x
# G*x<=h
# A*x=b
Q = 2*cvxopt.matrix([ [2, .5], [.5, 1] ])
p = cvxopt.matrix([1.0, 1.0])
G = cvxopt.matrix([[-1.0,0.0],[0.0,-1.0]])
h = cvxopt.matrix([0.0,0.0])
A = cvxopt.matrix([1.0, 1.0], (1,2))
b = cvxopt.matrix(1.0)
sol = cvxopt.solvers.qp(Q, p, G, h, A, b)
