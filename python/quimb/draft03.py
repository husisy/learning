'''link: https://quimb.readthedocs.io/en/latest/examples/ex_quench.html'''
import numpy as np
import quimb as qu
import matplotlib.pyplot as plt
plt.ion()

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

n = 18
H = qu.ham_heis(n, sparse=True).real
psi0 = qu.rand_product_state(n)

en_low, en_high = qu.bound_spectrum(H)
en_low, en_high, qu.expec(H, psi0)



def compute(t, pt):
    """Perform computation at time ``t`` with state ``pt``"""
    dims = [2] * n
    lns = [qu.logneg_subsys(pt, dims, i, i + 1) for i in range(n - 1)]
    mis = [qu.mutinf_subsys(pt, dims, i, i + 1) for i in range(n - 1)]
    return t, lns, mis

evo = qu.Evolution(psi0, H, compute=compute, progbar=True)
evo.update_to(5)

ts, lns, mis = zip(*evo.results)

fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(ts, lns)
axs[0].set_title("Logarithmic Negativity")
axs[1].plot(ts, mis)
axs[1].set_title("Mutual Information")


psi0 = qu.rand_product_state(n)
dims = [2 for _ in range(n)]
z0 = [qu.logneg_subsys(psi0, dims, x, x+1) for x in range(n-1)]
