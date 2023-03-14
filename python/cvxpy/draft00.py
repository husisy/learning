import cvxpy
import numpy as np

np_rng = np.random.default_rng()

cvxX = cvxpy.Variable()
cvxY = cvxpy.Variable()
cvxZ = cvxpy.Variable()

cvx0 = cvxpy.Variable((5, 4))
cvx0.shape #(5,4)
cvx0.size #20
cvx0.ndim #2
cvx0.sign #(str)UNKNOWN
z0 = np.random.rand(3,5) @ cvx0 #cvxpy.atoms.affine.binary_operators.MulExpression
z0.shape #(3,4)
z0.size #12
z0.ndim #2
z1 = cvxpy.Parameter(nonpos=True)
z1.sign #NONPOSITIVE

# parameter
cvxpy.Parameter(nonneg=True) # Positive scalar parameter.
cvxpy.Parameter(5) # Column vector parameter with unknown sign (by default).
z0 = cvxpy.Parameter((4, 7), nonpos=True) # Matrix parameter with negative entries.
z0.value = -np.ones((4, 7))
z1 = cvxpy.Parameter((4, 7), nonpos=True, value=-np.ones((4,7)))

# curvature
p0 = cvxpy.Parameter(nonneg=True)
cvxX.curvature #(str)AFFINE
p0.curvature #CONSTANT
cvxpy.square(cvxX).curvature #CONVEX
cvxpy.sqrt(cvxX).curvature #CONCAVE

# same expression, but cannot determine curvature for the former
cvxpy.sqrt(1 + cvxpy.square(cvxX)).curvature #QUASICONVEX
cvxpy.norm(cvxpy.hstack([1, cvxX]), 2).curvature #CONVEX

# https://www.cvxpy.org/tutorial/intro/index.html
constraints = [cvxX+cvxY==1, cvxX-cvxY>=1]
obj = cvxpy.Minimize((cvxX - cvxY)**2)
prob = cvxpy.Problem(obj, constraints)
prob.solve()
prob.status #(str)optimal
prob.value #optimal value
cvxX.value, cvxY.value #optimal variable
[x.dual_value for x in constraints]#optimal Lagrange multiplier for a constraint
# prob.constraints

prob_infeasible = cvxpy.Problem(cvxpy.Minimize(cvxX), [cvxX >= 1, cvxX <= 0])
prob_infeasible.solve() #inf
prob_infeasible.status #infeasible

prob_unbounded = cvxpy.Problem(cvxpy.Minimize(cvxX))
prob_unbounded.solve()
prob_unbounded.status #unbounded

# vector
m = 10
n = 5
A = np.random.randn(m, n)
b = np.random.randn(m)
x = cvxpy.Variable(n)
objective = cvxpy.Minimize(cvxpy.sum_squares(A @ x - b))
constraints = [0 <= x, x <= 1]
prob = cvxpy.Problem(objective, constraints)
prob.solve()



## semi-definite programming
n = 3
p = 3
C = np.random.randn(n, n)
A = [np.random.randn(n, n) for _ in range(p)]
b = [np.random.randn() for _ in range(p)]
# b = np.random.randn(p)
# A = np.random.randn(p, n, n)
X = cvxpy.Variable((n,n), symmetric=True)
# The operator >> denotes matrix inequality.
constraints = [X >> 0] + [cvxpy.trace(A[i] @ X) == b[i] for i in range(p)]
obj = cvxpy.Minimize(cvxpy.trace(C @ X))
prob = cvxpy.Problem(obj, constraints)
prob.solve()


## backward
p = cvxpy.Parameter()
x = cvxpy.Variable()
obj = cvxpy.Minimize(cvxpy.square(x - 2*p))
constraints = [x >= 0]
prob = cvxpy.Problem(obj, constraints)
p.value = 3.0
prob.solve(requires_grad=True, eps=1e-10) #x=2*p
prob.backward()
assert abs(p.gradient-2)<1e-7


## backward
b = cvxpy.Parameter()
x = cvxpy.Variable()
obj = cvxpy.Minimize(cvxpy.square(x - 2 * b))
constraints = [x >= 0]
prob = cvxpy.Problem(obj, constraints)
b.value = 3.
prob.solve(requires_grad=True, eps=1e-10)
x.gradient = 4.
prob.backward()
# dz/dp = dz/dx dx/dp = 4. * 2. == 8.
assert abs(b.gradient-8)<1e-7
