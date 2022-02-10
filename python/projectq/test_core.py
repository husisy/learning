import itertools
import opt_einsum
import numpy as np
import scipy.linalg
from collections import Counter

import projectq
from projectq import MainEngine
from projectq.ops import H, Measure, CNOT, All, QubitOperator, Rx, StatePreparation, FlipBits

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hf_bin_pad = lambda x,y: '{:>0{}b}'.format(x, y)


def np_random_initial_state(num_qubit):
    ret = np.random.randn(2**num_qubit) + np.random.randn(2**num_qubit)*1j
    return ret / np.sqrt(np.vdot(ret, ret))


def reverse_qubit(state):
    num_qubit = round(float(np.log2(state.size)))
    ret = state.reshape((2,)*num_qubit).transpose(*range(num_qubit-1,-1,-1)).reshape(-1)
    return ret


def qubit_str_sequence(num_qubit):
    ret = [''.join(x) for x in itertools.product(*['01' for _ in range(num_qubit)])] #000 001 010 ...
    return ret

def np_apply_single_gate(state, operator, target_qubit):
    num_state = len(state)
    num_qubit = round(float(np.log2(num_state)))
    assert num_state==(2**num_qubit)
    tmp0 = list(range(num_qubit))
    tmp0[target_qubit] = num_qubit
    ret = opt_einsum.contract( #TODO replace with opt_einsum
        state.reshape([2 for _ in range(num_qubit)]),
        list(range(num_qubit)),
        operator,
        [num_qubit,target_qubit],
        tmp0,
    ).reshape(-1)
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
    ret = opt_einsum.contract(tmp0, tmp1, tmp2, tmp3+qubit_sequence, tmp5).reshape(-1)
    return ret


def rx_gate(theta):
    tmp0 = np.cos(theta/2)
    tmp1 = np.sin(theta/2)
    ret = np.array([[tmp0,-1j*tmp1], [-1j*tmp1,tmp0]])
    return ret


def fetch_projectq_wavefunction(engine, id_list):
    engine.flush()
    mapping,wavefunction = engine.backend.cheat()
    assert set(id_list)==set(mapping.keys())
    tmp0 = [(len(id_list)-1-mapping[x]) for x in id_list]
    ret = np.array(wavefunction).reshape(*[2 for _ in range(len(id_list))]).transpose(*tmp0).reshape(-1)
    return ret


def fetch_projectq_all_amplitude(engine, q0):
    engine.flush()
    num_qubit = len(q0)
    tmp0 = [''.join(x) for x in itertools.product('01', repeat=num_qubit)]
    tmp1 = [engine.backend.get_amplitude(x, q0) for x in tmp0]
    ret0 = {x:y for x,y in zip(tmp0,tmp1)}
    ret1 = np.array(tmp1)
    return ret0,ret1


def test_projectq_wavefunction00():
    theta = np.random.uniform(0, 2*np.pi, 3)
    engine = MainEngine()
    q0 = engine.allocate_qureg(3)
    Rx(theta[0]) | q0[2]
    Rx(theta[1]) | q0[0]
    Rx(theta[2]) | q0[1]
    CNOT | (q0[0],q0[2])
    engine.flush()
    _,ret_ = fetch_projectq_all_amplitude(engine, q0)
    ret = fetch_projectq_wavefunction(engine, [x.id for x in q0])
    All(Measure) | q0
    assert hfe(ret_, ret) < 1e-7


def test_projectq_wavefunction01():
    theta = np.random.uniform(0, 2*np.pi, 3)
    ret_ = np.zeros(2**3)
    ret_[0] = 1
    ret_ = np_apply_single_gate(ret_, rx_gate(theta[0]), 2)
    ret_ = np_apply_single_gate(ret_, rx_gate(theta[1]), 0)
    ret_ = np_apply_single_gate(ret_, rx_gate(theta[2]), 1)

    eng = MainEngine()
    q0 = eng.allocate_qureg(3)
    Rx(theta[0]) | q0[2]
    Rx(theta[1]) | q0[0]
    Rx(theta[2]) | q0[1]
    eng.flush()
    ret = fetch_projectq_wavefunction(eng, [x.id for x in q0])
    All(Measure) | q0
    assert hfe(ret_, ret) < 1e-7


def test_StatePreparation(num_qubit=3):
    np0 = np_random_initial_state(num_qubit)
    eng = MainEngine()
    q0 = eng.allocate_qureg(num_qubit)
    StatePreparation(reverse_qubit(np0)) | q0
    ret = fetch_projectq_wavefunction(eng, [x.id for x in q0])
    All(Measure) | q0
    assert hfe(np0, ret) < 1e-7


def test_projectq_xgate(num_qubit=3):
    ind0 = np.random.randint(0, num_qubit)
    np0 = np_random_initial_state(num_qubit)
    XGate = np.array([[0,1],[1,0]])
    np_state = np_apply_single_gate(np0, XGate, ind0)
    engine = MainEngine()
    q0 = engine.allocate_qureg(num_qubit)
    StatePreparation(reverse_qubit(np0)) | q0
    projectq.ops.XGate() | q0[ind0]
    ret = fetch_projectq_wavefunction(engine, [x.id for x in q0])
    All(Measure) | q0
    assert hfe(np_state, ret) < 1e-7



def test_flip_bits_minus(num_qubit=3):
    XGate = np.array([[0,1],[1,0]])
    ind0 = np.random.randint(-num_qubit, num_qubit)
    np0 = np_random_initial_state(num_qubit)
    np_state = np0.copy()
    if ind0>=0:
        tmp0 = [x for x,y in enumerate(hf_bin_pad(ind0, num_qubit)[::-1]) if y=='1']
    else:
        tmp0 = [x for x,y in enumerate(hf_bin_pad(-ind0-1, num_qubit)[::-1]) if y=='0']
    for ind1 in tmp0:
        np_state = np_apply_single_gate(np_state, XGate, ind1)
    eng = MainEngine()
    q0 = eng.allocate_qureg(num_qubit)
    StatePreparation(reverse_qubit(np0)) | q0
    FlipBits(ind0) | q0
    ret = fetch_projectq_wavefunction(eng, [x.id for x in q0])
    All(Measure) | q0
    assert hfe(np_state, ret)<1e-7


def np_ccx_gate(num_control, operator):
    assert isinstance(num_control, int) and num_control>0
    tmp0 = (np.eye(2) for x in range(2**num_control-1))
    ret = scipy.linalg.block_diag(*tmp0, operator)
    return ret


def np_apply_ccx_gate(state, num_control, operator, qubit_sequence):
    ret = state.astype(np.find_common_type([state.dtype,operator.dtype], [])).reshape(2**num_control,-1)
    ret[-1] = np_apply_gate(ret[-1], operator, [x-num_control for x in qubit_sequence])
    ret = ret.reshape(-1)
    return ret


def test_projectq_CXGate(num_qubit=3):
    num_control = num_qubit - 1
    bit_sequence = np.random.permutation(num_qubit).tolist()
    np0 = np_random_initial_state(num_qubit)
    operator = np_ccx_gate(num_control, np.array([[0,1],[1,0]]))
    np_state = np_apply_gate(np0, operator, bit_sequence)
    engine = MainEngine()
    q0 = engine.allocate_qureg(num_qubit)
    StatePreparation(reverse_qubit(np0)) | q0
    tmp0 = [q0[x] for x in bit_sequence[:-1]]
    projectq.ops.C(projectq.ops.X, num_control) | (tmp0,q0[bit_sequence[-1]])
    ret = fetch_projectq_wavefunction(engine, [x.id for x in q0])
    All(Measure) | q0
    assert hfe(np_state, ret) < 1e-7


def test_projectq_CXGate01(num_qubit=6, num_control=3):
    theta = np.random.uniform(0, 2*np.pi)
    ind0 = np.random.randint(num_control, num_qubit)
    np0 = np_random_initial_state(num_qubit)
    operator = rx_gate(theta)
    np_state = np_apply_ccx_gate(np0, num_control, operator, [ind0])
    engine = MainEngine()
    q0 = engine.allocate_qureg(num_qubit)
    StatePreparation(reverse_qubit(np0)) | q0
    projectq.ops.C(projectq.ops.Rx(theta), num_control) | (q0[:num_control],q0[ind0])
    ret = fetch_projectq_wavefunction(engine, [x.id for x in q0])
    All(Measure) | q0
    assert hfe(np_state, ret) < 1e-7
