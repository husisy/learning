import itertools
import numpy as np

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

def hf_qubit_str_to_index(num_qubit, str0, return_bool=True):
    assert set(str0) <= set('01?')
    num_state = 2**num_qubit
    ret = np.ones(num_state, dtype=np.bool)
    for ind0,x in enumerate(str0):
        if x=='0':
            tmp0 = np.zeros(num_state, dtype=np.bool).reshape(2**ind0, 2, -1)
            tmp0[:,0] = True
            np.logical_and(ret, tmp0.reshape(-1), out=ret)
        elif x=='1':
            tmp0 = np.zeros(num_state, dtype=np.bool).reshape(2**ind0, 2, -1)
            tmp0[:,1] = True
            np.logical_and(ret, tmp0.reshape(-1), out=ret)
    if not return_bool:
        ret = np.nonzero(ret)[0]
    return ret


def qubit_str_sequence(num_qubit):
    ret = [''.join(x) for x in itertools.product(*['01' for _ in range(num_qubit)])] #000 001 010 ...
    return ret


def np_random_initial_state(num_qubit, tag_complex=True):
    ret = np.random.randn(2**num_qubit)
    if tag_complex:
        ret = ret + np.random.randn(2**num_qubit)*1j
    return ret / np.sqrt(np.vdot(ret, ret))


def generate_unitary_matrix(N0, tag_complex=True):
    if tag_complex:
        tmp0 = np.random.randn(N0, N0) + 1j*np.random.rand(N0,N0)
        _,ret = np.linalg.eig(tmp0 + np.conjugate(tmp0.T))
    else:
        tmp0 = np.random.randn(N0, N0)
        _,ret = np.linalg.eig(tmp0 + tmp0.T)
    return ret


def np_str_to_initial_state(str0):
    num_qubit = len(str0)
    ind0 = hf_qubit_str_to_index(num_qubit, str0, return_bool=False)[0]
    ret = np.zeros(2**num_qubit)
    ret[ind0] = 1
    return ret


def np_apply_gate(state, operator, qubit_sequence):
    # qubit_sequence: count from left to right |0123>
    num_state = len(state)
    num_qubit = round(float(np.log2(num_state)))
    N0 = len(qubit_sequence)
    assert num_state==(2**num_qubit)
    assert all(isinstance(x,int) and (0<=x) and (x<num_qubit) for x in qubit_sequence)
    assert len(qubit_sequence)==len(set(qubit_sequence))
    assert operator.ndim==2 and operator.shape[0]==operator.shape[1]
    assert operator.shape[0]==2**N0
    tmp0 = state.reshape([2 for _ in range(num_qubit)])
    tmp1 = list(range(num_qubit))
    tmp2 = operator.reshape([2 for _ in range(2*N0)])
    tmp3 = list(range(num_qubit,num_qubit+N0))
    tmp4 = {x:y for x,y in zip(qubit_sequence,tmp3)}
    tmp5 = [(tmp4[x] if x in tmp4 else x) for x in range(num_qubit)]
    ret = np.einsum(tmp0, tmp1, tmp2, tmp3+qubit_sequence, tmp5, optimize=True).reshape(-1)
    return ret


def U3_gate(a, theta, phi):
    ca = np.cos(a)
    sa = np.sin(a)
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    tmp0 = ca + 1j*sa*ct
    tmp3 = ca - 1j*sa*ct
    tmp1 = sa*st*(sp + 1j*cp)
    tmp2 = sa*st*(-sp + 1j*cp)
    ret = np.stack([tmp0,tmp1,tmp2,tmp3], axis=-1).reshape(*tmp0.shape, 2, 2)
    return ret


def np_inner_product_psi0_O_psi1(psi0, psi1, operator_list):
    '''
    psi0(np,?,(N0,))
    psi1(np,?,(N0,))
    operator_list(list,(tuple,%,2))
        %0(float): coefficient
        %1(list,(tuple,%,2))
            %0(np,?,(N1,N1))
            %1(tuple,int)
    operator_list(NoneType) TODO
    '''
    psi0_conjugate = np.conjugate(psi0)
    ret = 0
    for coefficient,term_i in operator_list:
        tmp_psi1 = psi1 #no need to copy
        for x,y in term_i:
            tmp_psi1 = np_apply_gate(tmp_psi1, x, y)
        ret = ret + coefficient * np.dot(psi0_conjugate, tmp_psi1)
    return ret
