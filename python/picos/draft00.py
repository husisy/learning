import numpy as np

import picos

np_rng = np.random.default_rng()
hf_randc = lambda *x: np_rng.uniform(-1,1,size=x) + 1j*np_rng.uniform(-1,1,size=x)


x = picos.RealVariable('x', 5)
a = picos.Constant('a', range(5))
P = picos.Problem()
P.minimize = abs(x-a)
P += picos.sum(x)==1
opt = P.solve(solver='cvxopt')
P.value #float
x.value  #cvxopt.base.matrix
x.np


P = picos.Problem()
x = picos.IntegerVariable('x', 2)
P += (2*x <= 11)
P.maximize = picos.sum(x)
P.solve(solver='glpk')
P.value

picos.BinaryVariable
picos.IntegerVariable
picos.RealVariable


## complex
# https://picos-api.gitlab.io/picos/complex.html
z = picos.ComplexVariable('z', 4) #4x1
z.value = [1,2+2j,3+3j,4j]
z.np
z.real
z.imag
z.conj
z.H
z.dim #8
~z #Convert between a valued expression and its value
z.value * z.H.value
A = (~z) * (~z.H)
A.hermitian #True

H = picos.HermitianVariable('H', 4) #4x4
H.dim #16
x = (H | A) #Tr[H^dagger A]
x.isreal #True
x = (H | A).refined
H >> 0 #PSD
picos.constraints.ComplexLMIConstraint.RealConversion.convert(H>>0, picos.Options()).get_constraint(0)
# <8×8 LMI Constraint: [Re(H), -Im(H); Im(H), Re(H)] ≽ 0>


## phase recovery in signal processing, phase-cut problem
tmp0 = hf_randc(5, 4)
M = picos.Constant("M", tmp0@tmp0.T.conj()) #rank-deficient hermitian matrix M
prob = picos.Problem()
U = picos.HermitianVariable("U", M.shape[0])
prob.set_objective("min", (U | M).real)
prob.add_constraint(picos.maindiag(U) == 1)
prob.add_constraint(U >> 0)
prob.solve(solver="cvxopt")
rank = (np.linalg.eigvalsh(U.np) > 1e-6).sum()
