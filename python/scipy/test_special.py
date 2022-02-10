import numpy as np
import scipy
import scipy.special

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


def test_erf():
    np_rng = np.random.default_rng()
    np0 = np_rng.uniform(-2, 2, size=(5,))
    eps = 1e-5
    ret0 = (scipy.special.erf(np0+eps) - scipy.special.erf(np0-eps)) / (2*eps)
    ret_ = 2/np.sqrt(np.pi) * np.exp(-np0**2)
    assert hfe(ret_, ret0) < 1e-5


def test_bessel():
    # https://en.wikipedia.org/wiki/Bessel_function
    np_rng = np.random.default_rng()
    eps = 1e-4
    order = np_rng.uniform(1, 2)
    x = np.linspace(1, 2, 100)

    # x^2*y'' + x*y' + (x^2-alpha^2)*y = 0
    tmp0 = [
        (lambda x: scipy.special.jv(order, x)), #first kind bessel function
        (lambda x: scipy.special.yv(order, x)), #second kind bessel function
    ]
    for hf0 in tmp0:
        d2y = (hf0(x-eps) + hf0(x+eps) - 2*hf0(x))/eps**2
        dy = (hf0(x+eps)-hf0(x-eps))/(2*eps)
        y = hf0(x)
        ret0 = x**2 * d2y + x*dy + (x**2-order**2)*y
        assert np.abs(ret0).max() < 1e-5

    # x^2*y'' + x*y' - (x^2+alpha^2)*y = 0
    tmp0 = [
        (lambda x: scipy.special.iv(order, x)), #modified bessel function of first kind
        (lambda x: scipy.special.kv(order, x)), #modified bessel function of second kind
    ]
    for hf0 in tmp0:
        d2y = (hf0(x-eps) + hf0(x+eps) - 2*hf0(x))/eps**2
        dy = (hf0(x+eps)-hf0(x-eps))/(2*eps)
        y = hf0(x)
        ret0 = x**2 * d2y + x*dy - (x**2+order**2)*y
        assert np.abs(ret0).max() < 1e-5


def test_bessel_root():
    np_rng = np.random.default_rng()
    n = np_rng.integers(0, 20)
    num0 = 10
    bessel_root = scipy.special.jn_zeros(n, num0)
    tmp0 = scipy.special.jv(n, bessel_root)
    assert np.abs(tmp0).max() < 1e-7
