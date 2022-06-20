# https://www.cvxpy.org/tutorial/intro/index.html
import cvxpy
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# Disciplined Parametrized Programming (DPP), trade-off curve
n = 15
m = 10
A = np.random.randn(n, m)
b = np.random.randn(n)
gamma = cvxpy.Parameter(nonneg=True) # gamma must be nonnegative due to DCP rules
x = cvxpy.Variable(m)
error = cvxpy.sum_squares(A @ x - b)
obj = cvxpy.Minimize(error + gamma*cvxpy.norm(x, 1))
prob = cvxpy.Problem(obj)

sq_penalty = []
l1_penalty = []
x_values = []
gamma_vals = np.logspace(-4, 6)
for val in gamma_vals:
    gamma.value = val
    prob.solve()
    sq_penalty.append(error.value)
    l1_penalty.append(cvxpy.norm(x, 1).value)
    x_values.append(x.value)

fig,(ax0,ax1) = plt.subplots(1, 2, figsize=(10,6))
ax0.plot(l1_penalty, sq_penalty)
ax0.set_xlabel(r'$\|x\|_1$')
ax0.set_ylabel(r'$\|Ax-b\|^2$')
ax0.set_title('Trade-Off Curve for LASSO')
for i in range(m):
    ax1.plot(gamma_vals, [xi[i] for xi in x_values])
ax1.set_xlabel(r'$\gamma$')
ax1.set_ylabel(r'$x_{i}$')
ax1.set_xscale('log')
ax1.set_title(r'Entries of x vs. $\gamma$')
fig.tight_layout()

