import cvxpy
import torch
import numpy as np
import cvxpylayers.torch

n, m = 2, 3
x = cvxpy.Variable(n)
A = cvxpy.Parameter((m, n))
b = cvxpy.Parameter(m)
constraints = [x >= 0]
objective = cvxpy.Minimize(0.5 * cvxpy.pnorm(A @ x - b, p=1))
problem = cvxpy.Problem(objective, constraints)
assert problem.is_dpp()

cvxpylayer = cvxpylayers.torch.CvxpyLayer(problem, parameters=[A, b], variables=[x])
A_tch = torch.randn(m, n, requires_grad=True)
b_tch = torch.randn(m, requires_grad=True)
solution, = cvxpylayer(A_tch, b_tch)
solution.sum().backward()



### PREAMBLE
# Differentiable Convex Optimization Layers
# CVXPY creates powerful new PyTorch and TensorFlow layers
# Akshay Agrawal, Brandon Amos, Shane Barratt, Stephen Boyd, Steven Diamond, J. Zico Kolter
# wideimg: ./overview.png

_x = cvxpy.Parameter(n)
_y = cvxpy.Variable(n)
obj = cvxpy.Minimize(cvxpy.sum_squares(_y-_x))
cons = [_y >= 0]
prob = cvxpy.Problem(obj, cons)
layer = cvxpylayers.torch.CvxpyLayer(prob, parameters=[_x], variables=[_y])
x = torch.linspace(-5, 5, steps=n, requires_grad=True)
y, = layer(x) #relu
y.sum().backward() #Heaviside


# sigmoid
_x = cvxpy.Parameter(n)
_y = cvxpy.Variable(n)
obj = cvxpy.Minimize(-_x.T*_y - cvxpy.sum(cvxpy.entr(_y) + cvxpy.entr(1.-_y)))
prob = cvxpy.Problem(obj)
layer = cvxpylayers.torch.CvxpyLayer(prob, parameters=[_x], variables=[_y])
x = torch.linspace(-5, 5, steps=n, requires_grad=True)
y, = layer(x)


# softmax
d = 5
_x = cvxpy.Parameter(d)
_y = cvxpy.Variable(d)
obj = cvxpy.Minimize(-_x.T*_y - cvxpy.sum(cvxpy.entr(_y)))
cons = [np.ones(d, dtype=np.float32).T*_y == 1.]
prob = cvxpy.Problem(obj, cons)
layer = cvxpylayers.torch.CvxpyLayer(prob, parameters=[_x], variables=[_y])
x = torch.randn(d, requires_grad=True)
y, = layer(x)


# sparsemax https://arxiv.org/abs/1602.02068
n = 4
x = cvxpy.Parameter(n)
y = cvxpy.Variable(n)
obj = cvxpy.sum_squares(x-y)
cons = [cvxpy.sum(y) == 1, 0. <= y, y <= 1.]
prob = cvxpy.Problem(cvxpy.Minimize(obj), cons)
layer = cvxpylayers.torch.CvxpyLayer(prob, [x], [y])
x = torch.randn(n)
y, = layer(x)


# csoftmax
n, k = 4, 2
u = torch.full([n], 1./k)
x = cvxpy.Parameter(n)
y = cvxpy.Variable(n)
obj = -x*y-cvxpy.sum(cvxpy.entr(y))
cons = [cvxpy.sum(y) == 1., y <= u]
prob = cvxpy.Problem(cvxpy.Minimize(obj), cons)
layer = cvxpylayers.torch.CvxpyLayer(prob, [x], [y])
x = torch.randn(n)
y, = layer(x)


# csparsemax
n, k = 4, 2
u = torch.full([n], 1./k)
x = cvxpy.Parameter(n)
y = cvxpy.Variable(n)
obj = cvxpy.sum_squares(x-y)
cons = [cvxpy.sum(y) == 1., 0. <= y, y <= u]
prob = cvxpy.Problem(cvxpy.Minimize(obj), cons)
layer = cvxpylayers.torch.CvxpyLayer(prob, [x], [y])
x = torch.randn(n)
y, = layer(x)


# Limited Multi-Label (LML) projection layer
n, k = 4, 2
x = cvxpy.Parameter(n)
y = cvxpy.Variable(n)
obj = -x*y-cvxpy.sum(cvxpy.entr(y))-cvxpy.sum(cvxpy.entr(1.-y))
cons = [cvxpy.sum(y) == k]
prob = cvxpy.Problem(cvxpy.Minimize(obj), cons)
layer = cvxpylayers.torch.CvxpyLayer(prob, [x], [y])
x = torch.randn(n)
y, = layer(x)


## OptNet QP
n, m, p = 10, 5, 5
Q_sqrt = cvxpy.Parameter((n, n))
q = cvxpy.Parameter(n)
A = cvxpy.Parameter((m, n))
b = cvxpy.Parameter(m)
G = cvxpy.Parameter((p, n))
h = cvxpy.Parameter(p)
x = cvxpy.Variable(n)
obj = cvxpy.Minimize(0.5*cvxpy.sum_squares(Q_sqrt*x) + q.T @ x)
cons = [A @ x == b, G @ x <= h]
prob = cvxpy.Problem(obj, cons)
layer = cvxpylayers.torch.CvxpyLayer(prob, parameters=[Q_sqrt, q, A, b, G, h], variables=[x])

torch.manual_seed(0)
Q_sqrtval = torch.randn(n, n, requires_grad=True)
qval = torch.randn(n, requires_grad=True)
Aval = torch.randn(m, n, requires_grad=True)
bval = torch.randn(m, requires_grad=True)
Gval = torch.randn(p, n, requires_grad=True)
hval = torch.randn(p, requires_grad=True)
y = layer(Q_sqrtval, qval, Aval, bval, Gval, hval)[0]
print('Output: ', y)
