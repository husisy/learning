import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import kwant

lat_chain = kwant.lattice.chain()

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hf_hermite = lambda x: (x+x.T.conj())/2
hf_randc = lambda *size: np.random.randn(*size) + 1j*np.random.randn(*size)


def test_band():
    N0 = 3
    for k_sign in [-1, 1]:
        dev0 = kwant.Builder(kwant.TranslationalSymmetry((k_sign,)))
        ham00 = hf_hermite(hf_randc(N0, N0))
        ham01 = hf_randc(N0, N0)
        dev0[lat_chain(0)] = ham00
        dev0[kwant.builder.HoppingKind((-1,), lat_chain, lat_chain)] = ham01
        dev0_f = kwant.wraparound.wraparound(dev0).finalized()
        kx = np.random.randn()
        ret0 = dev0_f.hamiltonian_submatrix(params={'k_x':kx})
        ret_ = ham00 + ham01*np.exp(1j*k_sign*kx) + ham01.T.conj()*np.exp(-1j*k_sign*kx)
        assert hfe(ret_, ret0) < 1e-10
