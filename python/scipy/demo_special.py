import numpy as np
import scipy
import scipy.special
import matplotlib.pyplot as plt
plt.ion()

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

# see wiki
def demo_airy():
    np0 = np.linspace(-2, 2)
    Ai,Aip,Bi,Bip = scipy.special.airy(np0)
    fig,ax = plt.subplots()
    ax.plot(np0, Ai, label='Ai')
    ax.plot(np0, Aip, label='Aip')
    ax.plot(np0, Bi, label='Bi')
    ax.plot(np0, Bip, label='Bip')
    ax.legend()
    ax.grid()

    zero_eps = 1e-5
    tmp0 = (scipy.special.airy(np0+zero_eps)[1] - scipy.special.airy(np0-zero_eps)[1])/(2*zero_eps)
    tmp1 = np0*Ai
    assert hfe(tmp0, tmp1) < 1e-7
    tmp0 = (scipy.special.airy(np0+zero_eps)[3] - scipy.special.airy(np0-zero_eps)[3])/(2*zero_eps)
    tmp1 = np0*Bi
    assert hfe(tmp0,tmp1) < 1e-7


def demo_parabolic_cylinder_function():
    # https://en.wikipedia.org/wiki/Parabolic_cylinder_function
    a = np.random.uniform(-1,  1)
    np0 = np.linspace(-1, 1, 100)
    zero_eps = 1e-4

    # f'' + (a + 0.5 - 0.25*x*x)*f=0
    for hf0 in [scipy.special.pbdv,scipy.special.pbvv]:
        fval, grad = hf0(a, np0)
        fval_plus,_ = hf0(a, np0+zero_eps)
        fval_minus,_ = hf0(a, np0-zero_eps)
        grad_ = (fval_plus-fval_minus) / (2*zero_eps)
        assert np.abs(grad_-grad).max() < 1e-5
        tmp0 = (fval_plus+fval_minus - 2*fval)/(zero_eps**2)
        tmp1 = (-0.25*np0**2 + a + 0.5)*fval
        assert np.abs(tmp0+tmp1).max() < 1e-5
    fval0,_ = scipy.special.pbdv(a, np0)
    fval1,_ = scipy.special.pbvv(a, np0)
    fig,ax = plt.subplots()
    ax.plot(np0, fval0, label='pbdv')
    ax.plot(np0, fval1, label='pbvv')
    ax.legend()
    ax.grid()

    # f'' + (0.25*x*x - a) = 0
    fval, grad = scipy.special.pbwa(a, np0)
    fval_plus,_ = scipy.special.pbwa(a, np0+zero_eps)
    fval_minus,_ = scipy.special.pbwa(a, np0-zero_eps)
    grad_ = (fval_plus-fval_minus) / (2*zero_eps)
    assert np.abs(grad_-grad).max() < 1e-5
    tmp0 = (fval_plus+fval_minus - 2*fval)/(zero_eps**2)
    tmp1 = (0.25*np0**2 - a)*fval
    assert np.abs(tmp0 + tmp1).max() < 1e-5

def demo_confluent_hypergeometric_function():
    # https://en.wikipedia.org/wiki/Confluent_hypergeometric_function
    a,b = np.random.uniform(0, 1, size=2)
    np0 = np.linspace(0.1, 0.9, 100)
    zero_eps = 1e-4

    solution_list = [
        scipy.special.hyp1f1, #Kummer function
        (lambda a,b,x: x**(1-b)*scipy.special.hyp1f1(1+a-b, 2-b, x)),
    ]
    for hf0 in solution_list:
        fval = hf0(a, b, np0)
        fval_plus = hf0(a, b, np0+zero_eps)
        fval_minus = hf0(a, b, np0-zero_eps)
        tmp0 = np0 * ((fval_plus+fval_minus - 2*fval)/(zero_eps**2))
        tmp1 = (b - np0) * (fval_plus-fval_minus) / (2*zero_eps) - a*fval
        assert np.abs(tmp0 + tmp1).max() < 1e-5
