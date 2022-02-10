import opt_einsum
import numpy as np
import pytenet as ptn

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


def test_as_vector(num_qubit_dim=3, num_qubit=5):
    num_bond_dim = [1] + np.random.randint(1, 7, num_qubit-1).tolist() + [1]
    mps = ptn.MPS(
        np.zeros(num_qubit_dim, dtype=np.int_),
        [np.zeros(x, dtype=np.int_) for x in num_bond_dim],
        fill='random',
    )
    tmp0 = [(2*x+1,2*x,2*x+2) for x in range(len(mps.A))]
    tmp1 = (y for x in zip(mps.A,tmp0) for y in x)
    tmp2 = list(range(1,2*len(mps.A),2))
    ret_ = opt_einsum.contract(*tmp1, tmp2).reshape(-1)
    assert hfe(ret_, mps.as_vector()) < 1e-7



num_qubit_dim = 3
num_qubit = 5
num_bond_dim = [1] + np.random.randint(1, 7, num_qubit-1).tolist() + [1]
mps = ptn.MPS(
    np.zeros(num_qubit_dim, dtype=np.int_),
    [np.zeros(x, dtype=np.int_) for x in num_bond_dim],
    fill='random',
)
[x.shape for x in mps.A]
mps.nsites
mps.bond_dims
mps.as_vector()
mps.orthonormalize(mode='left')



# MPO Hamiltonian
num_qubit_dim = 3
num_qubit = 5
E_hopping = 1
E_interaction = 4
E_chemical_potential = -0.5
BH = ptn.bose_hubbard_MPO(num_qubit_dim, num_qubit, E_hopping, E_interaction, E_chemical_potential)
[x.shape for x in BH.A]
BH.as_matrix()
tmp0 = [(3*x+1,3*x+2,3*x,3*x+3) for x in range(len(BH.A))]
tmp1 = (y for x in zip(BH.A,tmp0) for y in x)
tmp2 = [x[0] for x in tmp0] + [x[1] for x in tmp0]
ret = opt_einsum.contract(*tmp1, tmp2).reshape(num_qubit_dim**num_qubit, -1)
assert hfe(BH.as_matrix(), ret) < 1e-7


# MPO
PauliX = np.array([[0,1], [1,0]])
PauliY = np.array([[0,-1j], [1j,0]])
PauliZ = np.array([[1,0], [0,-1]])
eye2 = np.eye(2)

num_qubit_dim = 2
num_qubit = 5
opchain = ptn.OpChain(oplist=[PauliX, PauliX, PauliZ], istart=2, qD=[0, 0])
opchain.oplist
opchain.istart
opchain.qD
ret_ = opchain.as_matrix(num_qubit_dim, num_qubit)
ret = opt_einsum.contract(eye2,[0,1],eye2,[2,3],PauliX,[4,5],PauliX,[6,7],PauliZ,[8,9],[0,2,4,6,8,1,3,5,7,9])
assert hfe(ret_, ret.reshape(num_qubit_dim**num_qubit,-1)) < 1e-7

num_qubit_dim = 2
num_qubit = 5
opchain = ptn.OpChain(oplist=[PauliX, PauliX, PauliZ], istart=2, qD=[0, 0])
opchain2 = ptn.OpChain(oplist=[PauliY, PauliZ], istart=1, qD=[0])
opchain3 = ptn.OpChain(oplist=[PauliX], istart=4, qD=[])
mpo = ptn.MPO.from_opchains([0]*num_qubit_dim, num_qubit, [opchain, opchain2, opchain3])

def la_ji_OpChain(oplist, istart=0):
    assert len(oplist)>0
    ret = ptn.OpChain(oplist, [0]*(len(oplist)-1), istart)
    return ret

num_qubit_dim = 2
num_qubit = 3
opchain1 = la_ji_OpChain([PauliX, PauliX], istart=0)
opchain2 = la_ji_OpChain(oplist=[PauliY, PauliZ], istart=0)
opchain3 = la_ji_OpChain(oplist=[PauliY, PauliZ], istart=1)
mpo = ptn.MPO.from_opchains([0]*num_qubit_dim, num_qubit, [opchain1, opchain2])
mpo = ptn.MPO.from_opchains([0]*num_qubit_dim, num_qubit, [opchain1, opchain2])
