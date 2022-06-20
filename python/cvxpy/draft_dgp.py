# https://www.cvxpy.org/tutorial/dgp/index.html
# disciplined geometric programming
import cvxpy
import numpy as np

# DGP requires Variables to be declared positive via `pos=True`.
cvxX = cvxpy.Variable(pos=True)
cvxY = cvxpy.Variable(pos=True)
cvxZ = cvxpy.Variable(pos=True)

# log-log curvature
x0 = cvxpy.Constant(2.0)
x0.log_log_curvature #LOG-LOG CONSTANT
x1 = x0 * cvxX * cvxY #LOG-LOG AFFINE
x2 = x1 + (cvxX ** 1.5) * (cvxY ** -1) #LOG-LOG CONVEX
x3 = x2 ** -1 #LOG-LOG CONCAVE
x4 = x3 + x2 #UNKNOWN
# .is_log_log_constant()
# .is_log_log_affine()

obj = cvxpy.Maximize(cvxX * cvxY * cvxZ)
constraints = [4*cvxX*cvxY*cvxZ + 2*cvxX*cvxZ <= 10, cvxX <= 2*cvxY, cvxY <= 2*cvxX, cvxZ >= 1]
problem = cvxpy.Problem(obj, constraints)
problem.solve(gp=True)
