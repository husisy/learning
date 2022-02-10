import numpy as np
import orthopy
import sympy

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hf_legval_n = lambda x, n: np.polynomial.legendre.legval(x, np.array([0]*n+[1]))

N0 = 5
x = 0.233
ret_ = np.array([hf_legval_n(x, n) for n in range(N0)])
evaluator = orthopy.c1.legendre.Eval(x, scaling='classical') #iterator
ret0 = np.array([next(evaluator) for _ in range(N0)])
assert hfe(ret_, ret0) < 1e-7
# scaling=monic, the leading coefficient is 1
# scaling=classical, the maximum value is 1
# scaling=normal, the integral of the squared function over the domain is 1


x = sympy.Symbol('x')
evaluator = orthopy.c1.legendre.Eval(x, scaling='classical')
for _ in range(5):
    print(sympy.expand(next(evaluator)))


hf_np_kernel = lambda x: x/(np.exp(x)-1)
ret_ = scipy.integrate.quad(hf_np_kernel, 0, np.inf)[0]
ret0 = scipy.integrate.quad(hf_np_kernel, 0, 8)[0]
# int_0^8
