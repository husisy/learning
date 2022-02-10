import os
import pickle
import sympy
import numpy as np
import scipy.integrate
import quadpy
import orthopy

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

def hf_fermi_moment_scipy(k):
    hf0 = lambda x: x**(k+1)/(np.exp(x)-1)
    ret = scipy.integrate.quad(hf0, 0, np.inf)[0]
    return ret

hf_fermi_moment_sympy = lambda k: sympy.zeta(k+2) * sympy.factorial(k+1)


def solve_orthopy_point_weight(N0):
    assert isinstance(N0,int) and N0>1
    if N0>=9:
        print(f'WARNING, solve_orthopy_point_weight(N0={N0}) will spend many days to retrieve the results')
    moments = [hf_fermi_moment_sympy(x) for x in range(2*N0)]
    alpha, beta, int_1 = orthopy.tools.chebyshev(moments)
    alpha = [float(x) for x in alpha]
    beta = [float(x) for x in beta]
    points, weights = quadpy.tools.scheme_from_rc(alpha, beta, float(int_1))
    ret = {'alpha':alpha, 'beta':beta, 'points':points, 'weights':weights}
    return ret


def load_orthopy_point_weight(N0):
    filepath = 'fermi_quadrature_weight.pkl'
    assert isinstance(N0,int) and N0>1
    if not os.path.exists(filepath):
        with open(filepath, 'wb') as fid:
            pickle.dump({}, fid) #write empty dict first
    with open(filepath, 'rb') as fid:
        z0 = pickle.load(fid)
    if N0 not in z0:
        ret = solve_orthopy_point_weight(N0)
        z0[N0] = ret
        with open(filepath, 'wb') as fid:
            pickle.dump(z0, fid)
    else:
        ret = z0[N0]
    return ret['points'],ret['weights']


def test_fermi_moment():
    for k in range(5):
        assert hfe(hf_fermi_moment_scipy(k), hf_fermi_moment_sympy(k)) < 1e-5


def demo_fermi_quadrature():
    N0 = 6
    polynomial_coefficient = np.random.rand(N0)*2 - 1
    hf_poly = lambda x, polynomial: sum(z*(x**y) for y,z in enumerate(polynomial[::-1]))

    hf0 = lambda x: x/(np.exp(x)-1) * hf_poly(x,polynomial_coefficient)
    ret_,scipy_err = scipy.integrate.quad(hf0, 0, np.inf)

    points,weights = load_orthopy_point_weight(N0)
    ret0 = sum(hf_poly(x,polynomial_coefficient)*y for x,y in zip(points,weights))
    assert hfe(ret_, ret0) < 1e-10


def demo_approximated_by_exp():
    hf_kernel0 = lambda x,alpha: np.exp(-x) * (x**alpha)
    hf_kernel1 = lambda x: x/(np.exp(x) - 1)
    hf_poly = lambda x, polynomial: sum(z*(x**y) for y,z in enumerate(polynomial[::-1]))

    N0 = 8
    polynomial_coefficient = np.random.rand(N0//2)*2 - 1

    hf0 = lambda x: hf_poly(x, polynomial_coefficient) * hf_kernel1(x)
    ret_,scipy_err = scipy.integrate.quad(hf0, 0, np.inf)
    assert scipy_err < 1e-7

    hf1 = lambda x: hf_poly(x, polynomial_coefficient) * hf_kernel1(x) / hf_kernel0(x,alpha=0)
    scheme = quadpy.e1r.gauss_laguerre(2*N0)
    ret0 = scheme.integrate(hf1)[0]
    assert hfe(ret_, ret0) < 1e-7

    points,weights = load_orthopy_point_weight(N0)
    ret1 = sum(hf_poly(x,polynomial_coefficient)*y for x,y in zip(points,weights))
    assert hfe(ret_, ret1) < 1e-11
