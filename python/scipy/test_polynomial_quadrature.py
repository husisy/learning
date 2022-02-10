import numpy as np
import scipy.integrate

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


def legendre_quadrature(hf0, a, b, num_order):
    '''
    link: https://en.wikipedia.org/wiki/Gaussian_quadrature
    link: https://numpy.org/doc/1.18/reference/generated/numpy.polynomial.legendre.leggauss.html

    TODO functools.lru_cache https://docs.python.org/dev/library/functools.html#functools.lru_cache
    warning: root of high order Legendre Polynomial is unstable https://math.stackexchange.com/questions/2636801
    '''
    node,weight = np.polynomial.legendre.leggauss(num_order)
    x = node*(b-a)/2 + (b+a)/2
    ret = (b-a)/2 * np.sum(weight*hf0(x))
    return ret


def test_legendre_quadrature_node(num_order=10):
    node = np.polynomial.legendre.legroots(np.array([0]*num_order+[1]))
    tmp0 = np.polynomial.legendre.legval(node, np.array([0]*(num_order-1)+[1]))
    weight = 2*(1-node**2) / (num_order*tmp0)**2
    node_,weight_ = np.polynomial.legendre.leggauss(num_order)
    assert hfe(node_, node) < 1e-7
    assert hfe(weight_, weight) < 1e-7


def test_legendre_quadrature(num_order=10):
    hf0 = lambda x: (3-x-x**2)*np.sin(x)**2
    a = -1 - np.random.rand()/2
    b = 1 + np.random.rand()/2
    tmp0 = scipy.integrate.quad(hf0, a, b)
    assert tmp0[1] < 1e-5
    ret = legendre_quadrature(hf0, a, b, num_order)
    assert hfe(tmp0[0], ret)<1e-7


def chebyshev_quadrature(hf0, num_order, kind):
    '''
    link: https://en.wikipedia.org/wiki/Chebyshev%E2%80%93Gauss_quadrature
    warning: bad precision for manually adding factor sqrt(1-x**2)
    '''
    assert kind in {'first', 'second'}
    if kind=='first':
        node,weight = np.polynomial.chebyshev.chebgauss(num_order)
    else:
        node = np.cos(np.arange(1,num_order+1)*np.pi/(num_order+1))
        weight = np.pi/(num_order+1) * np.sin(np.arange(1,num_order+1)*np.pi/(num_order+1))**2
    ret = np.sum(weight*hf0(node))
    return ret


def test_chebyshev_quadrature(num_order=10):
    hf0 = lambda x: (3-x-x**2)*np.sin(x)**2
    a = -1 - np.random.rand()/2
    b = 1 + np.random.rand()/2
    hf1 = lambda x: hf0((b-a)*x/2 + (b+a)/2) #normalize the integral area to [-1,1]

    tmp0 = scipy.integrate.quad(lambda x: hf1(x)/np.sqrt(1-x**2), -1, 1)
    assert tmp0[1] < 1e-5
    ret = chebyshev_quadrature(hf1, num_order, kind='first')
    assert hfe(tmp0[0],ret) < 1e-7

    tmp0 = scipy.integrate.quad(lambda x: hf1(x)*np.sqrt(1-x**2), -1, 1)
    assert tmp0[1] < 1e-5
    ret = chebyshev_quadrature(hf1, num_order, kind='second')
    assert hfe(tmp0[0],ret) < 1e-7


def test_chebyshev_quadrature_node(num_order=10):
    node = np.cos(np.arange(1, 2*num_order, 2)*np.pi/(2*num_order))
    weight = (np.pi/num_order)*np.ones(num_order)
    node_,weight_ = np.polynomial.chebyshev.chebgauss(num_order)
    assert hfe(node_, node) < 1e-7
    assert hfe(weight_, weight) < 1e-7
