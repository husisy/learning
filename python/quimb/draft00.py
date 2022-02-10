import numpy as np
import quimb as qu
import matplotlib.pyplot as plt
plt.ion()

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

x0 = qu.qu([1,2j,-3], qtype='bra') #normalized=False
x0 = qu.bra([1,2j,-3])
x1 = qu.qu([1,2j,-3], qtype='ket', normalized=True)
x1 = qu.ket([1,2j,-3], normalized=True)
x2 = qu.qu([[1,2j,-3],[-2j,4,-6j],[-3,6j,9]], qtype='dop') #operator, sparse=False
x2 = qu.dop([[1,2j,-3],[-2j,4,-6j],[-3,6j,9]])

x2.H #dagger

qu.expectation(x1, x1) #inner product
qu.expectation(x1, x2)
qu.expectation(x2, x2) #trace


qu.bell_state('psi-') #see wiki
qu.up()
qu.down()
qu.pauli('X')
qu.pauli('Y')
qu.pauli('Z')
qu.hadamard()
qu.controlled('not')
qu.phase_gate(0.3)

qu.rand_ket(233)
qu.rand_herm(233)


qu.kron
qu.ikron(qu.pauli('X'), [2,2,2], inds=[1,])
qu.pkron
qu.partial_trace

# SLQ
tmp0 = qu.rand_rho(2**8) #is a sub-type of np.ndarray, just apply np.eig to verify the correctness
ret_ = np.sum(np.sqrt(np.linalg.eigvalsh(np.array(tmp0))))
ret = qu.tr_sqrt_approx(tmp0)

# automatic exploitation of symmetries via blocking
H = qu.ham_heis(12)
# el, ev = qu.eigh(H)
# el, ev = qu.eigh(H, autoblock=True)

# tensor network
ket = qu.tensor.Tensor(data=qu.bell_state('psi-').reshape(2,2), inds=('k0','k1'), tags={'KET'})
X = qu.tensor.Tensor(qu.pauli('X'), inds=('k0', 'b0'), tags={'PAULI', 'X', '0'})
Y = qu.tensor.Tensor(qu.pauli('Y'), inds=('k1', 'b1'), tags={'PAULI', 'Y', '1'})
bra = qu.tensor.Tensor(qu.rand_ket(4).reshape(2, 2), inds=('b0', 'b1'), tags={'BRA'})
TN = ket.H & X & Y & bra

# TN.graph(color=['KET', 'PAULI', 'BRA'], figsize=(4, 4))
# TN.graph(color=['KET', 'X', 'BRA', 'Y'], figsize=(4, 4))

TN ^ all #contract all, equivilent with np.einsum all
hf0 = lambda x: np.array(x.data)
np.einsum(hf0(ket.H), [0,1], hf0(X), [0,2], hf0(Y), [1,3], hf0(bra), [2,3], [])
TN ^ 'PAULI'
TN >> ['KET','X',('BRA','Y')] #cumulative contract
TN ^ ... #full structured contract
# TN ^ slice(100, 200) #fail??? contract of those sites only

# del TN['KET']

ket.split(left_inds=['k0'])

num_node = 10
tmp0 = [qu.tensor.Tensor() for _ in range(num_node)]
for ind0 in range(num_node):
    tmp0[ind0].new_ind(f'k{ind0}', size=2)
    tmp0[ind0].new_bond(tmp0[(ind0+1)%num_node], size=7)
mps = qu.tensor.TensorNetwork(tmp0)
mps.graph()

# with qu.tensor.contract_strategy('optimal')
# with qu.tensor.contract_backend('cupy')
# with qu.tensor.tensor_linop_backend('tensorflow')


## MPS and MPO
p = qu.tensor.MPS_rand_state(n=20, bond_dim=50)
print(p)
p.site_tag_id
p.site_ind_id
# p.show()
p.left_cannonize()
# p.show()
p.H @ p
(p.H & p).graph(color=[f'I{i}' for i in range(30)], initial_layout='random')
p2 = (p + p)/2
p2.compress(form=10)
p2[10] #p2['I10']
p2[10].H @ p2[10] #orthogonaility center?

A = qu.tensor.MPO_rand_herm(20, bond_dim=7, tags=['HAM'])
pH = p.H
p.align_(A, pH) #modifies the indices of each to form overlap
(pH & A & p).graph(color='HAM', iterations=1000)
(pH & A & p) ^ ... #contract everything, but use the structure if possible


## DRMG2
builder = qu.tensor.SpinHam(S=1)
builder += 1/2, '+', '-'
builder += 1/2, '-', '+'
builder += 1, 'Z', 'Z'
H = builder.build_mpo(n=100)
dmrg = qu.tensor.DMRG2(H, bond_dims=[10,20,100,100,200], cutoffs=1e-10) #OBC QBC
dmrg.solve(tol=1e-6, verbosity=1)
dmrg.state.show(max_width=80)


## Time Evolving Block Decimation (TEBD)
builder = qu.tensor.SpinHam(S=1/2)
builder.add_term(1, 'Z', 'Z')
builder.add_term(0.9, 'Y', 'Y')
builder.add_term(0.8, 'X', 'X')
builder.add_term(0.6, 'Z')
H = qu.tensor.NNI_ham_heis(20, bz=0.1)
H()
psi0 = qu.tensor.MPS_neel_state(20)
tebd = qu.tensor.TEBD(psi0, H)
tebd.update_to(T=3, tol=1e-3)
tebd.pt.show()

z0 = qu.tensor.MPS_neel_state(20)
np.stack([x.data.reshape(-1) for x in z0])
gate_Z = qu.pauli('Z')
[z0.gate(gate_Z,ind0).H @ z0 for ind0 in range(len(z0.tensors))]
