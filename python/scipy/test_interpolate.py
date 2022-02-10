import numpy as np
import scipy.interpolate
import scipy.sparse
import scipy.special

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


def my_pade_approximation(taylor_coefficient, q_order):
    '''f = p/q
    see wiki https://en.wikipedia.org/wiki/Pad%C3%A9_approximant
    also see https://github.com/husisy/PHYS4150-2018/blob/master/tutorial_material/TS20181011/TS20181011_pade.pdf

    taylor_coefficient(np,float,(N0,)): from high order to low
    q_order(int)
    (ret0)p_coefficient(np,float,(N0-q_order,)): from high order to low
    (ret1)q_coefficient(np,float,(q_order+1,)): from high order to low
    '''
    p_order = taylor_coefficient.size - 1 - q_order
    hf0 = lambda x,y: np.concatenate([np.ones(min(q_order, p_order+q_order-x))*y, np.zeros(max(0,x-p_order))])
    tmp0 = np.stack([hf0(x,y) for x,y in enumerate((-taylor_coefficient)[:0:-1])])
    block0 = scipy.sparse.spdiags(tmp0, -np.arange(1,p_order+q_order+1), p_order+q_order+1, q_order).todense()
    # matA = np.concatenate([np.eye(p_order+1), np.zeros(q_order)], axis=0)
    matA = np.block([[np.block([[np.eye(p_order+1)], [np.zeros((q_order,p_order+1))]]), block0]])
    tmp0 = np.linalg.solve(matA, taylor_coefficient[::-1])
    p_coefficient = tmp0[:p_order+1][::-1]
    q_coefficient = np.concatenate([tmp0[(p_order+1):][::-1], [1]])
    return p_coefficient,q_coefficient


def test_interpolate_pade():
    # from high order to low
    # exp_taylor = np.array([1/120, 1/24, 1/6, 1/2, 1, 1])
    taylor_coefficient = np.random.randn(8)
    # taylor_coefficient = np.array([1/120, 1/24, 1/6, 1/2, 1, 1])
    q_order = 3

    ret_p_, ret_q_ = my_pade_approximation(taylor_coefficient, q_order)

    tmp0 = scipy.interpolate.pade(taylor_coefficient[::-1], q_order)
    ret_p = tmp0[0].coefficients
    ret_q = tmp0[1].coefficients
    assert hfe(ret_p_, ret_p) < 1e-7
    assert hfe(ret_q_, ret_q) < 1e-7


def test_exponential_pade(taylor_order=9):
    exp_taylor = 1 / scipy.special.factorial(np.arange(taylor_order+1))[::-1]
    hfF = scipy.special.factorial
    hf_p_kmj = lambda k,m,j: hfF(k+m-j) * hfF(k) / (hfF(k+m) * hfF(k-j) * hfF(j))
    hf_q_kmj = lambda k,m,j: (-1)**j * hfF(k+m-j) * hfF(m) / (hfF(k+m) * hfF(m-j) * hfF(j))
    for q_order in range(taylor_order+1):
        p_order = taylor_order - q_order
        ret_p_ = np.array([hf_p_kmj(p_order,q_order,x) for x in range(p_order+1)])[::-1]
        ret_q_ = np.array([hf_q_kmj(p_order,q_order,x) for x in range(q_order+1)])[::-1]
        tmp0 = scipy.interpolate.pade(exp_taylor[::-1], q_order)
        ret_p = tmp0[0].coefficients
        ret_q = tmp0[1].coefficients
        assert hfe(ret_p_, ret_p) < 1e-7
        assert hfe(ret_q_, ret_q) < 1e-7
