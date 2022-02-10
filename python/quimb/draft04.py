'''link: https://quimb.readthedocs.io/en/latest/examples/ex_tn_rand_uni_gate_graphs.html'''
import numpy as np
import quimb as qu
import matplotlib.pyplot as plt
plt.ion()

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


## tensor network random unitary evolution
n = 50
cyclic = False
chi = 4  # intial bond dimension
psi = qu.tensor.MPS_rand_state(n, chi, cyclic=cyclic, tags='KET', dtype='complex128')

# the gates
n_gates = 5 * n
gates = [qu.rand_uni(4) for _ in range(n_gates)]
u_tags = [f'U{i}' for i in range(n_gates)]

for U, t in zip(gates, u_tags):
    # generate a random coordinate
    i = np.random.randint(0, n - int(not cyclic))

    # apply the next gate to the coordinate
    #     propagate_tags='sites' (the default in fact) specifies that the
    #     new gate tensor should inherit the site tags from tensors it acts on
    psi.gate_(U, where=[i, i + 1], tags=t, propagate_tags='sites')

psi.graph(color=['KET'])


fix = {
    # [key - tags that uniquely locate a tensor]: [val - (x, y) coord]
    **{('KET', f'I{i}'): (i, +10) for i in range(n)},
    # can also use a external index, 'k0' etc, as a key to fix it
    **{f'k{i}': (i, -10) for i in range(n)},
}
psi.graph(fix=fix, k=0.001, color=['I5', 'I15', 'I25', 'I35', 'I45'])



psiH = psi.H
psiH.retag_({'KET': 'BRA'})  # specify this to distinguish
norm = (psiH | psi)
norm.add_tag('UGs', where=u_tags, which='any')
norm.add_tag('VEC0', where=u_tags, which='!any')
norm.graph(color=['VEC0', 'UGs'])


fix = {
    **{(f'I{i}', 'KET', 'VEC0'): (i, -20) for i in range(n)},
    **{(f'I{i}', 'BRA', 'VEC0'): (i, +20) for i in range(n)},
}
(psiH | psi).graph(
    color=['VEC0', 'UGs', 'I5', 'I15', 'I25', 'I35', 'I45'],
    node_size=30,
    iterations=500,
    fix=fix, k=0.0001)


# this calculates an opimized path for the contraction, which is cached
#     the path can also be inspected with `print(expr)`
expr = (psi.H | psi).contract(all, get='path-info')
(psi.H | psi) ^ all


## manually perform partial trace
# make a 'bra' vector copy with 'upper' indices
psiH = psi.H
psiH.retag_({'KET': 'BRA'})
# this automatically reindexes the TN
psiH.site_ind_id = 'b{}'

# define two subsystems
sysa = range(15, 35)
sysb = [i for i in range(n) if i not in sysa]

# join indices for sysb only
psi.reindex_sites('dummy_ptr{}', sysb, inplace=True)
psiH.reindex_sites('dummy_ptr{}', sysb, inplace=True)

rho_ab = (psiH | psi)
fix = {
    **{f'k{i}': (i, -10) for i in range(n)},
    **{(f'I{i}', 'KET', 'VEC0'): (i, 0) for i in range(n)},
    **{(f'I{i}', 'BRA', 'VEC0'): (i, 10) for i in range(n)},
    **{f'b{i}': (i, 20) for i in range(n)},
}
rho_ab.graph(color=['VEC0'] + [f'I{i}' for i in sysa], iterations=500, fix=fix, k=0.001)
