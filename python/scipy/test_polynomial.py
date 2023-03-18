import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
# plt.ion()


def test_Polynomial(N0=3, N1=5):
    np_coefficient = np.random.rand(N0) #degree: 0,1,2,...,
    np_x = np.random.rand(N1)
    ret_ = np.sum(np_coefficient[:,np.newaxis] * (np_x**(np.arange(N0))[:,np.newaxis]), axis=0)
    ret0 = np.polynomial.Polynomial(np_coefficient)(np_x)
    assert hfe(ret_, ret0) < 1e-7

    np_root = np.random.rand(N0)
    ret0 = np.polynomial.Polynomial.fromroots(np_root)(np_x)
    ret_ = np.product(np_x-np_root[:,np.newaxis], axis=0)
    assert hfe(ret_, ret0) < 1e-7



def chebyshev2_chebval(x, coeff):
    '''
    see wiki: https://en.wikipedia.org/wiki/Chebyshev_polynomials

    U_n(x) = 2 Sum_{odd j} T_j(x) for odd n
    U_n(x) = 2 (Sum_{even j} T_j(x)) - 1 for even n
    '''
    coeff = np.asarray(coeff)
    cheb1_coeff = np.zeros_like(coeff)
    cheb1_coeff[0] = coeff[::2].sum()
    if coeff.size >= 1:
        cheb1_coeff[-1:0:-2] = 2*np.cumsum(coeff[-1:0:-2])
    if coeff.size >= 2:
        cheb1_coeff[-2:0:-2] = 2*np.cumsum(coeff[-2:0:-2])
    ret = np.polynomial.chebyshev.chebval(x, cheb1_coeff)
    return ret


hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hf_legval_n = lambda x, n: np.polynomial.legendre.legval(x, np.array([0]*n+[1]))
hf_chebval_n = lambda x, n: np.polynomial.chebyshev.chebval(x, np.array([0]*n+[1]))
hf_chebval2_n = lambda x,n: chebyshev2_chebval(x, np.array([0]*n+[1]))


def test_legendre_value():
    '''wiki: https://en.wikipedia.org/wiki/Legendre_polynomials'''
    tmp0 = [([1],1), ([1,0],1), ([3,0,-1],2), ([5,0,-3,0],2), ([35,0,-30,0,3],8),
            ([63,0,-70,0,15,0],8), ([231,0,-315,0,105,0,-5],16), ([429,0,-693,0,315,0,-35,0],16),
            ([6435,0,-12012,0,6930,0,-1260,0,35],128)]
    legendre_poly_list = [np.array(x)/y for x,y in tmp0]
    x = np.linspace(-1, 1, 100)
    for ind0,poly_i in enumerate(legendre_poly_list):
        hfe(np.polyval(poly_i,x), hf_legval_n(x, ind0)) < 1e-7


def test_legendre_orthogonal(N0=9):
    hf0 = lambda x,m,n: hf_legval_n(x,m)*hf_legval_n(x,n)
    tmp0 = [[scipy.integrate.quad(lambda x: hf0(x,m,n), -1, 1) for n in range(N0)] for m in range(N0)]
    relative_error = np.abs(np.array([[y[1] for y in x] for x in tmp0]))
    assert np.all(relative_error<1e-7)
    ret = np.array([[y[0] for y in x] for x in tmp0])
    ret_ = np.diag(2/(2*np.arange(N0)+1))
    assert hfe(ret_,ret) < 1e-6


def test_legendre_root(N0=8):
    hf_legroot_n = lambda n: np.polynomial.legendre.legroots(np.array([0]*n+[1]))
    for num0 in range(1, N0):
        tmp0 = hf_legroot_n(num0)
        assert np.max(np.abs(hf_legval_n(tmp0, num0))) < 1e-7


def plot_legendre():
    x = np.linspace(-2, 2, 100)
    fig,ax = plt.subplots(1,1)
    ax.plot(x, hf_legval_n(x, 1), label='L1')
    ax.plot(x, hf_legval_n(x, 2), label='L2')
    ax.plot(x, hf_legval_n(x, 3), label='L3')
    ax.plot(x, hf_legval_n(x, 4), label='L4')
    ax.legend()
    ax.set_title('legendre polynomial')
    ax.grid()
    ax.set_yscale('symlog')

    x = np.linspace(-2, 2, 100)
    fig,ax = plt.subplots(1,1)
    ax.plot(x, hf_legval_n(x, 5), label='L5')
    ax.plot(x, hf_legval_n(x, 6), label='L6')
    ax.plot(x, hf_legval_n(x, 7), label='L7')
    ax.plot(x, hf_legval_n(x, 8), label='L8')
    ax.legend()
    ax.set_title('legendre polynomial')
    ax.grid()
    ax.set_yscale('symlog')


def test_chebyshev1_value():
    '''first kind chebyshev see wiki: https://en.wikipedia.org/wiki/Chebyshev_polynomials'''
    cheby1_poly_list = [[1], [1,0], [2,0,-1], [4,0,-3,0], [8,0,-8,0,1], [16,0,-20,0,5,0],
            [32,0,-48,0,18,0,-1], [64,0,-112,0,56,0,-7,0], [128,0,-256,0,160,0,-32,0,1]]
    x = np.linspace(-1, 1, 100)
    for ind0,poly_i in enumerate(cheby1_poly_list):
        assert hfe(np.polyval(poly_i,x), hf_chebval_n(x, ind0)) < 1e-7


def test_chebyshev_orthogonal(N0=9):
    hf0 = lambda x,m,n: hf_chebval_n(x,m)*hf_chebval_n(x,n)/(np.pi*np.sqrt(1-x**2))
    tmp0 = [[scipy.integrate.quad(lambda x: hf0(x,m,n), -1, 1) for n in range(N0)] for m in range(N0)]
    relative_error = np.abs(np.array([[y[1] for y in x] for x in tmp0]))
    assert np.all(relative_error<1e-7)
    ret = np.array([[y[0] for y in x] for x in tmp0])
    ret_ = np.diag([1]+[0.5]*(N0-1))
    assert hfe(ret_,ret) < 1e-6


def test_chebyshev_root(N0=8):
    for num0 in range(1, N0):
        tmp0 = np.polynomial.chebyshev.chebroots(np.array([0]*num0+[1]))
        assert np.max(np.abs(hf_chebval_n(tmp0, num0))) < 1e-7
        tmp0 = np.cos(np.pi*(2*np.arange(num0)+1)/(2*num0))
        assert np.max(np.abs(hf_chebval_n(tmp0, num0))) < 1e-7


def plot_chebyshev():
    x = np.linspace(-5, 5, 100)
    fig,ax = plt.subplots(1,1)
    ax.plot(x, hf_chebval_n(x, 1), label='T1')
    ax.plot(x, hf_chebval_n(x, 2), label='T2')
    ax.plot(x, hf_chebval_n(x, 3), label='T3')
    ax.plot(x, hf_chebval_n(x, 4), label='T4')
    ax.legend()
    ax.set_title('chebyshev polynomial first kind')
    ax.grid()
    ax.set_yscale('symlog')

    x = np.linspace(-5, 5, 100)
    fig,ax = plt.subplots(1,1)
    ax.plot(x, hf_chebval_n(x, 5), label='T5')
    ax.plot(x, hf_chebval_n(x, 6), label='T6')
    ax.plot(x, hf_chebval_n(x, 7), label='T7')
    ax.plot(x, hf_chebval_n(x, 8), label='T8')
    ax.legend()
    ax.set_title('chebyshev polynomial first kind')
    ax.grid()
    ax.set_yscale('symlog')


def test_chebyshev2_value():
    '''second kind chebyshev wiki: https://en.wikipedia.org/wiki/Chebyshev_polynomials'''
    cheby2_poly_list = [[1], [2,0], [4,0,-1], [8,0,-4,0], [16,0,-12,0,1], [32,0,-32,0,6,0],
            [64,0,-80,0,24,0,-1], [128,0,-192,0,80,0,-8,0], [256,0,-448,0,240,0,-40,0,1]]
    x = np.linspace(-1, 1, 100)
    for ind0,poly_i in enumerate(cheby2_poly_list):
        assert hfe(np.polyval(poly_i,x), hf_chebval2_n(x, ind0)) < 1e-7


def test_chebyshev2_orthogonal(N0=9):
    hf0 = lambda x,m,n: hf_chebval2_n(x,m)*hf_chebval2_n(x,n)*(np.pi*np.sqrt(1-x**2))
    tmp0 = [[scipy.integrate.quad(lambda x: hf0(x,m,n), -1, 1) for n in range(N0)] for m in range(N0)]
    relative_error = np.abs(np.array([[y[1] for y in x] for x in tmp0]))
    assert np.all(relative_error<1e-7)
    ret = np.array([[y[0] for y in x] for x in tmp0])
    assert hfe(ret, np.pi**2/2*np.eye(N0)) < 1e-6


def test_chebyshe2_root(N0=8):
    for num0 in range(1, N0):
        tmp0 = np.cos(np.pi*(np.arange(num0)+1)/(num0+1))
        assert np.max(np.abs(hf_chebval2_n(tmp0, num0))) < 1e-7


def plot_chebyshev2():
    x = np.linspace(-5, 5, 100)
    fig,ax = plt.subplots(1,1)
    ax.plot(x, hf_chebval2_n(x, 1), label='U1')
    ax.plot(x, hf_chebval2_n(x, 2), label='U2')
    ax.plot(x, hf_chebval2_n(x, 3), label='U3')
    ax.plot(x, hf_chebval2_n(x, 4), label='U4')
    ax.legend()
    ax.set_title('chebyshev polynomial second kind')
    ax.grid()
    ax.set_yscale('symlog')

    x = np.linspace(-5, 5, 100)
    fig,ax = plt.subplots(1,1)
    ax.plot(x, hf_chebval2_n(x, 5), label='U5')
    ax.plot(x, hf_chebval2_n(x, 6), label='U6')
    ax.plot(x, hf_chebval2_n(x, 7), label='U7')
    ax.plot(x, hf_chebval2_n(x, 8), label='U8')
    ax.legend()
    ax.set_title('chebyshev polynomial second kind')
    ax.grid()
    ax.set_yscale('symlog')
