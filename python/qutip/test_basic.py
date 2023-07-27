import numpy as np
import scipy.linalg
import scipy.integrate

import qutip
import qutip_qip
import qutip_qip.circuit

np_rng = np.random.default_rng()

def rand_state(N0):
    tmp0 = np_rng.uniform(-1,1,size=N0) + 1j*np_rng.uniform(-1,1,size=N0)
    ret = tmp0/np.linalg.norm(tmp0)
    return ret

def rand_density_matrix(N0):
    tmp0 = rand_matrix(N0, N0)
    tmp1 = tmp0 @ tmp0.T.conj()
    ret = tmp1 / np.trace(tmp1)
    return ret


def rand_matrix(N0, N1=None):
    if N1 is None:
        N1 = N0
    ret = np_rng.uniform(-1,1,size=(N0,N0)) + 1j*np_rng.uniform(-1,1,size=(N0,N0))
    return ret


def rand_hermite_matrix(N0):
    tmp0 = rand_matrix(N0, N0)
    ret = tmp0 + tmp0.T.conj()
    return ret


def test_operator_and_vector():
    N0 = 5
    op0 = qutip.Qobj(rand_matrix(N0))
    ret_ = op0.full().T.reshape(-1)
    ret0 = qutip.operator_to_vector(op0).full()[:,0]
    assert np.abs(ret_-ret0).max() < 1e-10

    tmp0 = qutip.operator_to_vector(op0)
    ret_ = tmp0.full().reshape(*[x[0] for x in tmp0.dims[0]]).T
    ret0 = qutip.vector_to_operator(tmp0).full()
    assert np.abs(ret_-ret0).max() < 1e-10


def test_spre_spost():
    N0 = 3
    op0 = qutip.Qobj(rand_matrix(N0))
    ret_ = np.kron(np.eye(N0), op0.full())
    ret0 = qutip.spre(op0).full() #superoperator formed from pre-multiplication
    assert np.abs(ret_-ret0).max() < 1e-10

    ret_ = np.kron(op0.full().T, np.eye(N0))
    ret0 = qutip.spost(op0).full() #Superoperator formed from post-multiplication
    assert np.abs(ret_-ret0).max() < 1e-10

    ret_ = np.kron(op0.full().conj(), op0.full())
    ret0 = qutip.to_super(op0).full() #represent a quantum map to the supermatrix (Liouville) representation
    assert np.abs(ret_-ret0).max() < 1e-10

    op0 = qutip.Qobj(rand_matrix(N0))
    op1 = qutip.Qobj(rand_matrix(N0))
    ret_ = np.kron(op1.full().T, op0.full())
    ret0 = qutip.sprepost(op0, op1).full()
    assert np.abs(ret_-ret0).max() < 1e-10


def test_to_choi():
    N0 = 3
    op0 = qutip.Qobj(rand_matrix(N0))
    op1 = qutip.to_super(op0)
    ret_ = op1.full().reshape(N0,N0,N0,N0).transpose(3,1,2,0).reshape(N0**2,N0**2)
    ret0 = qutip.to_choi(op1).full()
    assert np.abs(ret_-ret0).max() < 1e-10


# TODO to_kraus
# TODO to_stinespring
# TODO to_chi


def test_sesolve():
    N0 = 3
    hamiltonian = qutip.Qobj(rand_hermite_matrix(N0))
    q0 = qutip.Qobj(rand_state(N0))
    tspan = np.linspace(0, 10, 20)
    operator = qutip.Qobj(rand_matrix(N0))

    ret2 = qutip.sesolve(hamiltonian, q0, tspan, [operator]).expect[0]

    ret_ = []
    EVL,EVC = np.linalg.eigh(hamiltonian.full())
    q0_np = q0.full()[:,0]
    op_np = operator.full()
    for time_i in tspan:
        tmp0 = (EVC * np.exp(-1j*EVL*time_i)) @ EVC.T.conj()
        tmp1 = tmp0 @ q0_np
        ret_.append(np.vdot(tmp1, op_np @ tmp1))
    ret_ = np.array(ret_)
    assert np.abs(ret2 - ret_).max() < 1e-4

    ## TODO compare with scipy.integrate.ode
    # def hf1(t, y, hamiltonian=hamiltonian.full()):
    #     return -1j * hamiltonian @ y
    # hf2 = lambda x, operator=operator.full(): (hfH(x[:,np.newaxis]) @ operator @ x[:,np.newaxis])[0,0]
    # tmp0 = hfH(state_initial.full()) @ operator.full() @ state_initial.full()
    # ret1 = [hf2(state_initial.full()[:,0])]

    # z0 = scipy.integrate.ode(hf1)
    # z0.set_integrator('zvode', method='adams')
    # z0.set_initial_value(time_list[0], state_initial.full()[:,0])
    # for time_i in time_list[1:]:
    #     if not z0.successful():
    #         break
    #     tmp0 = z0.integrate(time_i)
    #     ret1.append(hf2(tmp0))


def test_fidelity():
    N0 = 5
    q0 = qutip.Qobj(rand_state(N0))
    q1 = qutip.Qobj(rand_state(N0))
    ret_ = abs(np.vdot(q0.full()[:,0], q1.full()[:,0]))
    # ret_ = np.abs((q0.dag()*q1).full()[0,0])
    ret0 = qutip.fidelity(q0, q1)
    assert abs(ret_-ret0) < 1e-10

    dm0 = qutip.Qobj(rand_density_matrix(N0))
    dm1 = qutip.Qobj(rand_density_matrix(N0))
    tmp0 = scipy.linalg.sqrtm(dm0.full())
    tmp1 = np.linalg.eigvals(tmp0 @ dm1.full() @ tmp0).real
    ret_ = np.sqrt(tmp1[tmp1>0]).sum()
    ret0 = qutip.fidelity(dm0, dm1)
    assert abs(ret_-ret0) < 1e-10


def test_trace_distance():
    N0 = 5
    q0 = qutip.Qobj(rand_state(N0))
    q1 = qutip.Qobj(rand_state(N0))
    fidelity = abs(np.vdot(q0.full()[:,0], q1.full()[:,0]))
    ret_ = np.sqrt(1-fidelity**2)
    ret0 = qutip.tracedist(q0, q1)
    assert abs(ret_-ret0) < 1e-7 #strange accuracy
    # TODO density matrix


# TODO qutip.average_gate_fidelity


def test_3cnot_swap():
    # a swap gate is equivalent to three CNOT gates
    ret_ = qutip.operations.swap().full()
    z0 = qutip_qip.circuit.QubitCircuit(2)
    z0.add_gate('SWAP', targets=[0,1])
    ret0 = qutip_qip.operations.gate_sequence_product(z0.propagators())
    ret1 = qutip.operations.cnot(control=0, target=1) * qutip.operations.cnot(control=1, target=0) * qutip.operations.cnot(control=0, target=1)
    ret2 = qutip.operations.cnot(control=1, target=0) * qutip.operations.cnot(control=0, target=1) * qutip.operations.cnot(control=1, target=0)
    assert all(np.abs(ret_-x.full()).max() < 1e-10 for x in [ret0,ret1,ret2])


def test_toffoli_gate_decomposition():
    np0 = np.block([[np.eye(6),np.zeros((6,2))],[np.zeros((2,6)),np.array([[0,1],[1,0]])]])

    circ = qutip_qip.circuit.QubitCircuit(3, reverse_states=False)
    circ.add_gate('TOFFOLI', controls=[0,1], targets=[2])

    x0 = qutip_qip.operations.gate_sequence_product(circ.propagators())
    assert np.abs(np0 - x0.data.toarray()).max() < 1e-10

    circ1 = circ.resolve_gates(['CNOT','RX','RY','RZ'])
    x1 = qutip_qip.operations.gate_sequence_product(circ1.propagators())
    assert np.abs(np0 - x1.data.toarray()).max() < 1e-10


def test_teleportation():
    # https://nbviewer.org/urls/qutip.org/qutip-tutorials/tutorials-v4/quantum-circuits/teleportation.ipynb
    circ = qutip_qip.circuit.QubitCircuit(3, num_cbits=2, input_states=[r"\psi", "0", "0", "c0", "c1"])
    # alice: q0, q1
    # bob: q2

    # shared-EPR pair
    circ.add_gate('SNOT', targets=[1]) # Hadamard gate
    circ.add_gate('CNOT', controls=[1], targets=[2])

    circ.add_gate('CNOT', controls=[0], targets=[1])
    circ.add_gate('SNOT', targets=[0]) #Hadamard gate

    circ.add_measurement('M0', targets=[0], classical_store=1)
    circ.add_measurement('M0', targets=[1], classical_store=0)
    # 00->I 01->Z 10->X 11->ZX
    circ.add_gate('X', targets=[2], classical_controls=[0])
    circ.add_gate('Z', targets=[2], classical_controls=[1])

    tmp0 = np_rng.uniform(-1, 1, size=2) + 1j*np_rng.uniform(-1, 1, size=2)
    q0 = tmp0 / np.linalg.norm(tmp0)
    tmp0 = qutip.tensor(qutip.Qobj(q0[:,np.newaxis]), qutip.basis(2,0), qutip.basis(2,0))
    q1 = circ.run(tmp0)
    _,S,V = np.linalg.svd(q1.data.toarray().reshape(4, 2))
    assert S[1] < 1e-7
    assert abs(abs(np.vdot(V[0], q0)) - 1) < 1e-7
