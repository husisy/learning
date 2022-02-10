import sympy
import quadpy
import numpy as np
import scipy.integrate

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))

def test_quad_basic(N0=5):
    np_w = np.random.randn(N0)
    x0,x1 = np.sort(np.random.rand(2))
    def hf0(x):
        x = np.asarray(x)
        ret = np_w.reshape((N0,) + (1,)*x.ndim) * x
        return ret

    tmp0 = [scipy.integrate.quad(lambda x: hf0(x)[i], x0, x1) for i in range(N0)]
    ret_ = np.array([x[0] for x in tmp0])
    np_err = np.array([x[1] for x in tmp0])
    assert np.all(np_err < 1e-7)

    ret_ = np.array([scipy.integrate.quad(lambda x: hf0(x)[i], x0, x1)[0] for i in range(N0)])
    ret0,quadpy_err = quadpy.quad(hf0, x0, x1)
    assert np.all(quadpy_err < 1e-7)
    assert hfe(ret_, ret0) < 1e-7

    ret1,quadpy_err = quadpy.c1.integrate_adaptive(hf0, [x0,x1])
    assert np.all(quadpy_err < 1e-7)
    assert hfe(ret_, ret1) < 1e-7


def test_chebyshev_gauss_1_quadrature(N0=6):
    # N0 should be even in quadpy implementation
    polynomial_coefficient = np.random.rand(N0)*2 - 1

    hf0 = lambda x,_para=polynomial_coefficient: sum(z*(x**y) for y,z in enumerate(_para[::-1])) / np.sqrt(1-x**2)
    ret_,scipy_err = scipy.integrate.quad(hf0, -1, 1)
    assert scipy_err < 1e-5

    hf1 = lambda x,_para=polynomial_coefficient: sum(z*(x**y) for y,z in enumerate(_para[::-1]))
    scheme = quadpy.c1.chebyshev_gauss_1(N0)
    # scheme.show()
    ret0 = scheme.integrate(hf1, intervals=[-1,1])
    ret1 = np.sum(hf1(scheme.points)*scheme.weights)
    assert hfe(ret_, ret0) < 1e-5
    assert hfe(ret0, ret1) < 1e-7


def test_chebyshev_gauss_2_quadrature(N0=6):
    # N0 should be even in quadpy implementation
    polynomial_coefficient = np.random.rand(N0)*2 - 1

    hf0 = lambda x,_para=polynomial_coefficient: sum(z*(x**y) for y,z in enumerate(_para[::-1])) * np.sqrt(1-x**2)
    ret_,scipy_err = scipy.integrate.quad(hf0, -1, 1)
    assert scipy_err < 1e-5

    hf1 = lambda x,_para=polynomial_coefficient: sum(z*(x**y) for y,z in enumerate(_para[::-1]))
    scheme = quadpy.c1.chebyshev_gauss_2(N0)
    # scheme.show()
    ret0 = scheme.integrate(hf1, intervals=[-1,1])
    ret1 = np.sum(hf1(scheme.points)*scheme.weights)
    assert hfe(ret_, ret0) < 1e-5
    assert hfe(ret0, ret1) < 1e-7


def test_gauss_laguerre_quadrature(N0=5, alpha=None):
    if alpha is None:
        alpha = (np.random.rand()-0.5) * 2 * 0.9
    polynomial_coefficient = np.random.rand(N0)*2 - 1

    hf0 = lambda x,_para=polynomial_coefficient: np.exp(-x) * (x**alpha) * sum(z*(x**y) for y,z in enumerate(_para[::-1]))
    ret_,scipy_err = scipy.integrate.quad(hf0, 0, np.inf)
    assert scipy_err < 1e-5

    hf1 = lambda x,_para=polynomial_coefficient: sum(z*(x**y) for y,z in enumerate(_para[::-1]))
    scheme = quadpy.e1r.gauss_laguerre(N0, alpha=alpha)
    # scheme.show()
    ret0 = scheme.integrate(hf1)[0]
    ret1 = np.sum(hf1(scheme.points)*scheme.weights)
    assert hfe(ret_, ret0) < 1e-5
    assert hfe(ret0, ret1) < 1e-7
