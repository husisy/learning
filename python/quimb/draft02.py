import functools
import numpy as np
import quimb as qu

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

def hf_remove_phase(np0:np.ndarray):
    assert np0.ndim==1
    np1 = np.abs(np0)
    ind0 = np.argmax(np1)
    return np0 * np1[ind0] / np0[ind0]

def pure_state_dm_to_wave(density_matrix:np.ndarray):
    assert density_matrix.ndim==2
    assert np.max(np.abs(density_matrix - np.conjugate(density_matrix.T))) < 1e-7
    EVL,EVC = np.linalg.eigh(density_matrix)
    assert (abs(EVL[-1]-1) < 1e-7) and (np.max(EVL[:-1]) < 1e-7)
    return EVC[:,-1]

def test_product_state(N0=5):
    dims = [2 for _ in range(N0)]
    z0 = qu.rand_product_state(N0)
    ret_ = np.array(z0)[:,0]
    tmp0 = [pure_state_dm_to_wave(np.array(qu.partial_trace(z0, dims=dims, keep=[x]))) for x in range(N0)]
    ret = functools.reduce(np.kron, tmp0)
    assert hfe(hf_remove_phase(ret), hf_remove_phase(ret_)) < 1e-7


def my_partial_trace(psi0, dims, keep):
    assert np.prod(dims)==psi0.size
    N0 = len(dims)
    N1 = len(keep)
    assert (set(keep) < set(range(N0))) and (len(set(keep))==len(keep))
    ind0 = np.array(sorted(keep) + sorted(set(range(N0))-set(keep)))
    tmp0 = psi0.reshape(dims).transpose(*ind0)
    bra = np.conjugate(tmp0)
    ket = tmp0[tuple([slice(None)]*N1 + [None]*N1)]
    ret = (ket*bra).sum(axis=tuple(range(2*N1,N1+N0))).reshape(np.prod([dims[x] for x in keep]), -1)
    return ret

def test_partial_trace(N0=5):
    dims = np.random.randint(2, 4, size=(N0,))
    psi0 = qu.rand_ket(np.prod(dims))
    tmp0 = [[x] for x in range(N0)] + [[x,y] for x in range(N0) for y in range(N0) if x!=y]
    for ind0 in tmp0:
        ret_ = qu.partial_trace(psi0, dims=dims, keep=ind0)
        ret = my_partial_trace(psi0, dims, ind0)
        assert hfe(ret_, ret) < 1e-7


def my_ham_heis(N0, j=1, cyclic=True):
    sigma_x = np.array([[0,1],[1,0]]) / 2
    sigma_y = np.array([[0,-1j],[1j,0]]) / 2
    sigma_z = np.array([[1,0],[0,-1]]) / 2
    if N0==1:
        return j * (sigma_x+sigma_y+sigma_z)
    tmp0 = j*(np.kron(sigma_x,sigma_x) + np.kron(sigma_y,sigma_y) + np.kron(sigma_z,sigma_z)).real
    ret = 0
    for ind0 in range(N0-1):
        ret = ret + np.kron(np.kron(np.eye(2**ind0), tmp0), np.eye(2**(N0-2-ind0)))
    if cyclic:
        ret = ret + j*np.kron(np.kron(sigma_x, np.eye(2**(N0-2))), sigma_x).real
        ret = ret + j*np.kron(np.kron(sigma_y, np.eye(2**(N0-2))), sigma_y).real
        ret = ret + j*np.kron(np.kron(sigma_z, np.eye(2**(N0-2))), sigma_z).real
    return ret


def test_ham_heis(N0=4):
    ret_ = np.array(qu.ham_heis(N0, cyclic=True))
    ret = my_ham_heis(N0, cyclic=True)
    assert hfe(ret_,ret) < 1e-7

    ret_ = np.array(qu.ham_heis(N0, cyclic=False))
    ret = my_ham_heis(N0, cyclic=False)
    assert hfe(ret_,ret) < 1e-7


def test_evolution():
    hamiltonian = qu.pauli('Z')
    initial_state = qu.ket([1,1], normalized=True)
    sigma_x = qu.pauli('X')
    time_array = np.linspace(0,2*np.pi,100)
    ret_ = np.cos(2*time_array)
    ret = np.array([qu.expectation(x,sigma_x) for x in qu.Evolution(initial_state, hamiltonian).at_times(time_array)])
    assert hfe(ret_, ret) < 1e-7 #just solve the schrodinger equation
