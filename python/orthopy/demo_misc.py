import sympy
import numpy as np
import scipy.integrate
import quadpy
import orthopy

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

def custom_quadrature():
    # https://github.com/nschloe/orthopy/wiki/Generating-1D-recurrence-coefficients-for-a-given-weight#stieltjes
    # https://dx.doi.org/10.2307/2004418
    N = 4

    scheme = quadpy.c1.chebyshev_gauss_1(N)
    weights_ = scheme.weights.copy()
    points_ = scheme.points.copy()

    # Golub-Welsch
    x = sympy.Symbol("x")
    moments = [sympy.integrate(x**k/sympy.sqrt(1-x**2), (x, -1, +1)) for k in range(2*N + 1)]
    moments = [float(m) for m in moments]  # convert to floats
    alpha, beta, int_1 = orthopy.tools.golub_welsch(moments)
    points, weights = quadpy.tools.scheme_from_rc(alpha, beta, int_1)
    assert hfe(weights_, weights[::-1]*np.sqrt(np.pi)) < 1e-5
    assert hfe(points_, points[::-1]) < 1e-5

    # Stieltjes
    alpha, beta, int_1 = orthopy.tools.stieltjes(
        lambda x, fx: sympy.integrate(fx/sympy.sqrt(1-x**2), (x, -1, 1)),
        N,
    )
    points, weights = quadpy.tools.scheme_from_rc(alpha, beta, int_1, mode='sympy')
    points = np.array([float(x) for x in points])
    weights = np.array([float(x) for x in weights])
    assert hfe(weights_, weights[::-1]) < 1e-5
    assert hfe(points_, points[::-1]) < 1e-5

    # Chebyshev
    x = sympy.Symbol('x')
    moments = [sympy.integrate(x**k/sympy.sqrt(1-x**2), (x, -1, +1)) for k in range(2*N)]
    alpha, beta, int_1 = orthopy.tools.chebyshev(moments)
    points, weights = quadpy.tools.scheme_from_rc(alpha, beta, int_1, mode='sympy')
    points = np.array([float(x) for x in points])
    weights = np.array([float(x) for x in weights])
    assert hfe(weights_, weights[::-1]) < 1e-5
    assert hfe(points_, points[::-1]) < 1e-5
