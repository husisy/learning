'''link: https://quimb.readthedocs.io/en/latest/examples/ex_2d.html'''
import itertools
import numpy as np
import quimb as qu
import matplotlib.pyplot as plt
plt.ion()

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

def ham_heis_2D(n, m, j=1.0, sparse=True):
    dims = [[2 for _ in range(m)] for _ in range(n)]
    # generate tuple of all site coordinates
    sites = [(x,y) for x in range(n) for y in range(m)]

    # generate neighbouring pairs of coordinates
    def gen_pairs():
        for i, j, in sites:
            above, right = (i + 1) % n, (j + 1) % m
            if above != 0:
                yield ((i, j), (above, j))
            if right != 0:
                yield ((i, j), (i, right))

    # generate all pairs of coordinates and directions
    pairs_ss = tuple(itertools.product(gen_pairs(), 'xyz'))

    # make XX, YY and ZZ interaction from pair_s
    #     e.g. arg ([(3, 4), (3, 5)], 'z')
    def interactions(pair_s):
        pair, s = pair_s
        Sxyz = qu.spin_operator(s, sparse=True)
        return qu.ikron([j * Sxyz, Sxyz], dims, pair)

    H = sum(map(interactions, pairs_ss))

    if qu.isreal(H): #for speedup
        H = H.real
    if not sparse:
        H = qarray(H.A)
    return H


n, m = 4, 5
dims = [[2 for _ in range(m)] for _ in range(n)]
H = ham_heis_2D(n, m)
H = H + 0.2 * qu.ikron(qu.spin_operator('Z', sparse=True), dims, [(1, 2)])
ge, gs = qu.eigh(H, k=1) #almost 16 seconds
ge[0] #-11.661573929790721

Sz = qu.spin_operator('Z', stype='coo')
Sz_ij = [[qu.ikron(Sz, dims, [(i, j)]) for j in range(m)] for i in range(n)]
m_ij = [[qu.expec(Sz_ij[i][j], gs) for j in range(m)] for i in range(n)]
plt.imshow(m_ij)
plt.colorbar()

target = (1, 2)
rho_ab_ij = [[qu.partial_trace(gs, dims=dims, keep=[target, (i, j)]) for j in range(m)] for i in range(n)]
mi_ij = [[qu.mutinf(rho_ab_ij[i][j] if (i, j) != target else qu.purify(rho_ab_ij[i][j])) for j in range(m)] for i in range(n)]
plt.imshow(mi_ij)
plt.colorbar()

Sy = qu.spin_operator('y')
z_corr = qu.correlation(None, Sy, Sy, 0, 1, dims=[2, 2], precomp_func=True)
cy_ij = [[z_corr(rho_ab_ij[i][j] if (i, j) != target else qu.purify(rho_ab_ij[i][j])) for j in range(m)] for i in range(n)]
plt.imshow(cy_ij)
plt.colorbar()


