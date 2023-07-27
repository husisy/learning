import numpy as np

import qutip
import qutip.control.pulseoptim
import qutip.control.grape
import qutip_qip
import qutip_qip.circuit
import qutip_qip.algorithms
import qutip_qip.vqa
import qutip_qip.device

hf_unitary_dist = lambda x,y: abs(np.trace(x.full().T.conj() @ y.full())/x.shape[0])
np_rng = np.random.default_rng()

def demo_crab():
    # Calculation of control fields for state-to-state transfer of a 2 qubit system using CRAB algorithm
    # https://nbviewer.jupyter.org/github/qutip/qutip-notebooks/blob/master/examples/control-pulseoptim-CRAB-2qubitInerac.ipynb
    np_rng = np.random.default_rng(234)
    alpha = np_rng.uniform(0,1,size=2)
    beta = np_rng.uniform(0,1,size=2)

    s0 = qutip.qeye(2)
    sx = qutip.sigmax()
    sz = qutip.sigmaz()
    sxs0 = qutip.tensor(sx, s0)
    s0sx = qutip.tensor(s0, sx)
    szs0 = qutip.tensor(sz, s0)
    s0sz = qutip.tensor(s0, sz)
    szsz = qutip.tensor(sz, sz)

    H_d = alpha[0]*sxs0 + alpha[1]*s0sx + beta[0]*szs0 + beta[1]*s0sz
    H_c = [szsz]
    q_init = qutip.fock(4, 0)
    q_target = qutip.fock(4, 3)

    evolution_time = 18
    num_time_slice = 100
    pulse_type = 'DEF' # pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|

    # Nelder-Mead algorithm
    result = qutip.control.pulseoptim.opt_pulse_crab_unitary(H_d, H_c, q_init, q_target, num_time_slice, evolution_time,
                fid_err_targ=1e-3, max_iter=1000, max_wall_time=120,
                init_coeff_scaling=5.0, num_coeffs=5, method_params={'xtol':1e-3},
                guess_pulse_type=None, guess_pulse_action='modulate', gen_stats=True)
    result.fid_err #Final fidelity error
    result.evo_full_final #final state
    result.grad_norm_final #final gradient norm
    result.termination_reason
    result.num_iter
    # result.stats.report()
    result.initial_amps #(np,float64,(100,1))
    result.final_amps #(np,float64,(100,1))


def demo_grape_single_qubit_unitary():
    # GRAPE calculation of control fields for single-qubit rotation
    # https://nbviewer.jupyter.org/github/qutip/qutip-notebooks/blob/master/examples/control-grape-single-qubit-rotation.ipynb
    total_time = 1
    num_iteration = 150
    tspan = np.linspace(0, total_time, 50)
    theta, phi = np_rng.uniform(0, 1, size=2)
    U_target = qutip.operations.rz(phi) * qutip.operations.rx(theta)
    H_ops = [qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]
    H0 = 0 * np.pi * qutip.sigmaz()

    tmp0 = np_rng.uniform(0, np.pi*0.01, size=(len(H_ops), len(tspan)))
    tmp1 = np.ones(10)*0.1
    u0 = [np.convolve(tmp1, tmp0[x], mode='same') for x in range(len(H_ops))]
    # about 8 seconds
    result = qutip.control.grape.cy_grape_unitary(U_target, H0, H_ops, num_iteration, tspan, u_start=u0,
                eps=2*np.pi/total_time, phase_sensitive=False)
    # progress_bar=qutip.ui.progressbar.TextProgressBar()
    hf_unitary_dist(U_target, result.U_f)
    result.u #(np,(150,3,50))

    U_f_numerical = qutip.propagator(result.H_t, tspan, args={})[-1]
    hf_unitary_dist(U_target, U_f_numerical)


def demo_grape_iswap():
    # GRAPE calculation of control fields for iSWAP implementation
    # https://nbviewer.jupyter.org/github/qutip/qutip-notebooks/blob/master/examples/control-grape-iswap.ipynb
    total_time = 1
    tspan = np.linspace(0, total_time, 100)
    U_target = qutip.operations.iswap()
    num_iteration = 50
    H_ops = [
        qutip.tensor(qutip.sigmax(), qutip.sigmax()),
        qutip.tensor(qutip.sigmay(), qutip.sigmay()),
        qutip.tensor(qutip.sigmaz(), qutip.sigmaz())
    ]
    H0 = 0 * np.pi * (qutip.tensor(qutip.sigmaz(), qutip.qeye(2)) + qutip.tensor(qutip.qeye(2), qutip.sigmaz()))

    tmp0 = np_rng.uniform(0, np.pi*0.01, size=(len(H_ops), len(tspan)))
    tmp1 = np.ones(10)*0.1
    u0 = [np.convolve(tmp1, tmp0[x], mode='same') for x in range(len(H_ops))]
    result = qutip.control.grape.cy_grape_unitary(U_target, H0, H_ops, num_iteration, tspan, u_start=u0, eps=2*np.pi/total_time)
    hf_unitary_dist(U_target, result.U_f)

    U_f_numerical = qutip.propagator(result.H_t, tspan, [], args={})[-1]
    hf_unitary_dist(U_target, U_f_numerical)


def demo_grape_cnot():
    # GRAPE calculation of control fields for CNOT implementation
    # https://nbviewer.jupyter.org/github/qutip/qutip-notebooks/blob/master/examples/control-grape-cnot.ipynb
    num_iteration = 500
    total_time = 1
    tspan = np.linspace(0, total_time, 50)
    U_target = qutip.operations.cnot()

    H_ops = [
        qutip.tensor(qutip.sigmax(), qutip.qeye(2)),
        qutip.tensor(qutip.sigmay(), qutip.qeye(2)),
        qutip.tensor(qutip.sigmaz(), qutip.qeye(2)),
        qutip.tensor(qutip.qeye(2), qutip.sigmax()),
        qutip.tensor(qutip.qeye(2), qutip.sigmay()),
        qutip.tensor(qutip.qeye(2), qutip.sigmaz()),
        qutip.tensor(qutip.sigmax(),qutip.sigmax()) + qutip.tensor(qutip.sigmay(),qutip.sigmay()) + qutip.tensor(qutip.sigmaz(),qutip.sigmaz())
    ]
    H0 = 0 * np.pi * (qutip.tensor(qutip.sigmax(), qutip.qeye(2)) + qutip.tensor(qutip.qeye(2), qutip.sigmax()))

    tmp0 = np_rng.uniform(0, np.pi*0.01, size=(len(H_ops), len(tspan)))
    tmp1 = np.ones(10)*0.1
    u0 = [np.convolve(tmp1, tmp0[x], mode='same') for x in range(len(H_ops))]
    u_limits = None #[0, 1 * 2 * pi]
    alpha = None
    # about 40s
    result = qutip.control.grape.cy_grape_unitary(U_target, H0, H_ops, num_iteration, tspan, u_start=u0, u_limits=u_limits,
                eps=2*np.pi*1, alpha=alpha, phase_sensitive=False, progress_bar=qutip.ui.progressbar.TextProgressBar())
    hf_unitary_dist(U_target, result.U_f)

    U_f_numerical = qutip.propagator(result.H_t, tspan, [], options=qutip.Odeoptions(nsteps=500), args={})[-1]
    hf_unitary_dist(U_target, U_f_numerical)


def demo_qubit_circuit():
    # Quantum gates and circuits
    # https://nbviewer.jupyter.org/github/qutip/qutip-notebooks/blob/master/examples/quantum-gates.ipynb#Gates-in-QuTiP-and-their-representation
    # https://en.wikipedia.org/wiki/Quantum_logic_gate
    z0 = qutip_qip.circuit.QubitCircuit(3, reverse_states=False)
    z0.add_gate('RX', targets=[0], arg_value=np.pi/2) # rx(np.pi/2)
    z0.add_gate('RY', targets=[0], arg_value=np.pi/2) # ry(np.pi/2)
    z0.add_gate('RZ', targets=[0], arg_value=np.pi/2) # rz(np.pi/2)
    z0.add_gate('CNOT', controls=[0], targets=[1]) # cnot()
    z0.add_gate('CSIGN', controls=[0], targets=[1]) # csign() cphase(np.pi)
    z0.add_gate('CNOT', targets=1, controls=0)
    z0.add_gate('RX', targets=0, arg_value=np.pi/2)
    z0.add_gate('RY', targets=1, arg_value=np.pi/2)
    z0.add_gate('RZ', targets=2, arg_value=np.pi/2)
    z0.add_gate('ISWAP', targets=[1,2])
    matU = qutip_qip.operations.gate_sequence_product(z0.propagators())
    # .resolve_gates() #TODO

    qutip.operations.berkeley() #.add_gate('BERKELEY', targets=[0,1])
    qutip.operations.swap()
    qutip.operations.iswap()
    qutip.operations.swapalpha(np.pi/2)
    qutip.operations.sqrtiswap()
    qutip.operations.sqrtswap()
    qutip.operations.sqrtnot()
    qutip.operations.snot() #hadamard_transform()
    qutip.operations.phasegate(np.pi/2)
    qutip.operations.globalphase(np.pi/2)
    qutip.operations.fredkin()
    qutip.operations.toffoli()
    qutip.operations.cnot(N=3)


def demo_qft():
    num_qubit = 3
    circ = qutip_qip.algorithms.qft_gate_sequence(num_qubit, swapping=False, to_cnot=True)
    x0 = qutip_qip.operations.gate_sequence_product(circ.propagators())

    hf_fft_mat = lambda N0: np.exp(-2j*np.pi/N0*(np.arange(N0)[:,np.newaxis]*np.arange(N0))) / np.sqrt(N0)
    x1 = hf_fft_mat(2**num_qubit)
    np.abs(x0.data.toarray() - x1.T.conj()).max()

    np.linalg.eigvals(x0.data.toarray() @ x1.T.conj())


def demo_vqa():
    model = qutip_qip.vqa.VQA(num_qubits=1, num_layers=1, cost_method='OBSERVABLE')
    tmp0 = qutip_qip.vqa.VQABlock(qutip.sigmax()/2, name='R_x(\\theta)')
    model.add_block(tmp0)
    # model.export_image()
    model.cost_observable = qutip.sigmaz()
    result = model.optimize_parameters(method='L-BFGS-B', use_jac=True, bounds=[[0,4]])
    result.res.x[0] #3.14159


def demo_pauli_level_simulation():
    circ = qutip_qip.circuit.QubitCircuit(3)
    circ.add_gate('X', targets=2)
    circ.add_gate('SNOT', targets=0)
    circ.add_gate('SNOT', targets=1)
    circ.add_gate('SNOT', targets=2)

    ## oracle function f(x)
    circ.add_gate('CNOT', controls=0, targets=2)
    circ.add_gate('CNOT', controls=1, targets=2)

    circ.add_gate('SNOT', targets=0)
    circ.add_gate('SNOT', targets=1)

    ## gate-level simulation
    q0 = qutip.basis([2,2,2], [0,0,0])
    q1 = circ.run(q0) #ideal result

    ## pulse-level simulation (spin chain)
    proc = qutip_qip.device.LinearSpinChain(num_qubits=3, sx=0.25, t2=30)
    # sigma_x drive strength 0.25MHz
    proc.load_circuit(circ)
    # fig, ax = proc.plot_pulses(figsize=(8, 5))
    tspan = np.linspace(0, 20, 300)
    q2 = proc.run_state(q0, tlist=tspan)

    ## pulse-level simulation (superconducint qubits)
    proc = qutip_qip.device.SCQubits(num_qubits=3)
    proc.load_circuit(circ)
    # fig, ax = proc.plot_pulses(figsize=(8, 5))

    options = {
        "SNOT": {"num_tslots": 6, "evo_time": 2},
        "X": {"num_tslots": 1, "evo_time": 0.5},
        "CNOT": {"num_tslots": 12, "evo_time": 5},
    }
    model = qutip_qip.device.SpinChainModel(3, setup="linear")
    proc = qutip_qip.device.OptPulseProcessor(num_qubits=3, model=model)
    proc.load_circuit(circ, setting_args=options, merge_gates=False, verbose=True, amp_ubound=5, amp_lbound=0)
    # fig, ax = proc.plot_pulses(figsize=(8, 5))
    # proc.set_tlist
    # proc.set_coeffs
