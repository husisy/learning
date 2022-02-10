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
