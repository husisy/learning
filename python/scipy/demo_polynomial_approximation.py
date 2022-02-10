import functools
import numpy as np
import scipy.integrate
from numpy.polynomial import chebyshev
import matplotlib.pyplot as plt
plt.ion()

# see https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.78.275

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hf_chebval_n = lambda x, n: chebyshev.chebval(x, np.array([0]*n+[1]))
hf_gauss_delta = lambda x,sigma: 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-x**2/(2*sigma**2))
hf_cauchy_delta = lambda x,epsilon: epsilon/(np.pi*(x**2+epsilon**2))


hf_dirichlet_kernel = lambda x: np.ones(x)
hf_fejer_kernel = lambda x: 1 - np.arange(x)/x
def hf_jackson_kernel(x):
    tmp0 = np.arange(x)
    tmp1 = (x+1-tmp0)*np.cos(tmp0*np.pi/(x+1))
    tmp2 = np.sin(tmp0*np.pi/(x+1))/np.tan(np.pi/(x+1))
    ret = (tmp1 + tmp2) / (x+1)
    return ret
hf_lorentz_kernel = lambda x,lambda_: np.sinh(lambda_*(1-np.arange(x)/x)) / np.sinh(lambda_)


def get_kernel_list(kernel_name, kernel_para=None):
    kernel_para_ = {'lorentz_lambda': 4}
    if kernel_para is not None:
        kernel_para_.update(kernel_para_)
    assert len(kernel_name)>0 and all(isinstance(x,str) for x in kernel_name)
    kernel_name = set(x.lower() for x in kernel_name)
    assert kernel_name <= {'dirichlet', 'fejer', 'jackson', 'lorentz'}
    ret = []
    if 'dirichlet' in kernel_name:
        ret.append((hf_dirichlet_kernel, 'Dirichlet')),
    if 'fejer' in kernel_name:
        ret.append((hf_fejer_kernel, 'Fejer'))
    if 'jackson' in kernel_name:
        ret.append((hf_jackson_kernel, 'Jackson'))
    if 'lorentz' in kernel_name:
        tmp0 = functools.partial(hf_lorentz_kernel, lambda_=kernel_para_['lorentz_lambda'])
        tmp1 = r'lorentz($\lambda={}$)'.format(kernel_para_['lorentz_lambda'])
        ret.append((tmp0,tmp1))
    return ret


def demo00(hf0, order_list, hf_kernel):
    np_xdata = np.linspace(-1, 1, 502)[1:-1]
    tmp0 = [scipy.integrate.quad(lambda x: hf0(x)*hf_chebval_n(x,i), -1, 1) for i in range(max(order_list))]
    assert all(x[1]<1e-7 for x in tmp0)
    moments = [x[0] for x in tmp0]
    fig,ax = plt.subplots()
    ax.plot(np_xdata, hf0(np_xdata), label='f(x)')
    for order in order_list:
        tmp0 = np.array([moments[0]]+[2*x for x in moments[1:order]]) * hf_kernel(order)
        np_approximation = chebyshev.chebval(np_xdata, tmp0) / (np.pi*np.sqrt(1-np_xdata**2))
        ax.plot(np_xdata, np_approximation, label='cheby-{}'.format(order))
    ax.legend()
    ax.set_title('different order')


def demo00_delta(order_list, hf_kernel, delta_x=0):
    assert (-1<delta_x) and (delta_x<1)
    np_xdata = np.linspace(-1, 1, 502)[1:-1]
    moments = [hf_chebval_n(delta_x,i) for i in range(max(order_list))]
    fig,ax = plt.subplots()
    for order in order_list:
        tmp0 = np.array([moments[0]]+[2*x for x in moments[1:order]]) * hf_kernel(order)
        np_approximation = chebyshev.chebval(np_xdata, tmp0) / (np.pi*np.sqrt(1-np_xdata**2))
        ax.plot(np_xdata, np_approximation, label='cheby-{}'.format(order))
    ax.legend()
    ax.set_title('different order')


def demo01(hf0, order, kernel_list, lorentz_lambda=4):
    np_xdata = np.linspace(-1, 1, 502)[1:-1]
    tmp0 = [scipy.integrate.quad(lambda x: hf0(x)*hf_chebval_n(x,i), -1, 1) for i in range(order)]
    assert all(x[1]<1e-7 for x in tmp0)
    moments = [x[0] for x in tmp0]
    fig,ax = plt.subplots()
    ax.plot(np_xdata, hf0(np_xdata), 'x', label='f(x)')
    for hf_kernel,label in kernel_list:
        tmp0 = np.array([moments[0]]+[2*x for x in moments[1:]]) * hf_kernel(order)
        np_approximation = chebyshev.chebval(np_xdata, tmp0) / (np.pi*np.sqrt(1-np_xdata**2))
        ax.plot(np_xdata, np_approximation, label=label)
    ax.legend()
    ax.set_title('different kernel')


def demo01_delta(order, kernel_list=None, lorentz_lambda=4, delta_x=0):
    assert (-1<delta_x) and (delta_x<1)
    np_xdata = np.linspace(-1, 1, 502)[1:-1]
    moments = [hf_chebval_n(delta_x,i) for i in range(order)]
    fig,ax = plt.subplots()

    tmp0 = hf_gauss_delta(np_xdata[::5]-delta_x, np.pi/order)
    ax.plot(np_xdata[::5], tmp0, 'x', label=r'gauss($\sigma=\pi/{}$)'.format(order))

    tmp0 = hf_cauchy_delta(np_xdata[::5]-delta_x, lorentz_lambda/order)
    ax.plot(np_xdata[::5], tmp0, 'x', label=r'cauchy($\lambda={}/{}$)'.format(lorentz_lambda, order))

    for hf_kernel,label in kernel_list:
        tmp0 = np.array([moments[0]]+[2*x for x in moments[1:]]) * hf_kernel(order)
        np_approximation = chebyshev.chebval(np_xdata, tmp0) / (np.pi*np.sqrt(1-np_xdata**2))
        ax.plot(np_xdata, np_approximation, label=label)
    ax.legend()
    ax.set_title('different kernel')


# hf0 = lambda x: np.array(x>0, np.float)
hf0 = lambda x: np.array(x<0, np.float)
# hf0 = lambda x: np.sin(5*x)**2
# hf0 = lambda x: 1 / (1+x**2)
# hf0 = lambda x: np.array(x>-0.5, np.float) + np.array(x>0.5, np.float)
demo00(hf0, order_list=(15,30,45,60), hf_kernel=hf_jackson_kernel)
demo00_delta(order_list=(15,30,45,60), hf_kernel=hf_jackson_kernel)

# kernel_list = get_kernel_list(['dirichlet', 'fejer', 'jackson', 'lorentz'], kernel_para={'lorentz_lambda': 4})
kernel_list = get_kernel_list(['dirichlet', 'jackson', 'lorentz'], kernel_para={'lorentz_lambda': 4})
demo01(hf0, order=64, kernel_list=kernel_list)
demo01_delta(order=64, kernel_list=kernel_list)
