import numpy as np
import scipy.optimize

import qat.lang
import qat.qpus
import qat.plugins
import qat.core

np_rng = np.random.default_rng()

@qat.lang.qrout
def bell_pair():
    qat.lang.H(0)
    qat.lang.CNOT(0, 1)
result = bell_pair().run() #len(result)=2
for sample in result:
    print(f"{sample.state}: {sample.amplitude}")


qprog = qat.lang.Program()
qbits = qprog.qalloc(size=2) #number of qubits
qat.lang.H(qbits[0])
qat.lang.CNOT(qbits[0], qbits[1])
circ = qprog.to_circ()
qpu = qat.qpus.get_default_qpu()
result = qpu.submit(circ.to_job())
for sample in result:
    print(f"{sample.state}: {sample.amplitude}")




## This is a standard implementation of Grover's diffusion
@qat.lang.qrout(unroll=False)
def diffusion(k):
    for wire in range(2 * k):
        qat.lang.H(wire)
        qat.lang.X(wire)
    qat.lang.Z.ctrl(2 * k - 1)(list(range(2*k)))
    for wire in range(2 * k):
        qat.lang.X(wire)
        qat.lang.H(wire)

@qat.lang.qrout(unroll=False)
def diffusion01(k):
    with qat.lang.compute(): #automatically uncomputes at the end
        for wire in range(2 * k):
            qat.lang.H(wire)
            qat.lang.X(wire)
    qat.lang.Z.ctrl(2 * k - 1)(list(range(2*k)))

@qat.lang.qrout(unroll=False)
def is_palindrome(k):
    first_half = list(range(k))
    second_half = list(range(k, 2 * k))
    with qat.lang.compute():
        for w1, w2 in zip(first_half, reversed(second_half)):
            qat.lang.CNOT(w1, w2)
        for w2 in second_half:
            qat.lang.X(w2)
    qat.lang.Z.ctrl(k - 1)(second_half)

@qat.lang.qrout(unroll=False)
def grover(k):
    qbits = list(range(2 * k))
    diff = diffusion(k)
    oracle = is_palindrome(k)
    for qbit in qbits:
        qat.lang.H(qbit)
    nsteps = int(np.pi * np.sqrt(2 ** k) / 4)
    for _ in range(nsteps):
        oracle(qbits)
        diff(qbits)

result = grover(k=2).run()
for sample in result:
    print(sample.state, sample.probability)



@qat.lang.qrout
def bell_pair():
    qat.lang.H(0)
    qat.lang.CNOT(0, 1)
result = qat.qpus.get_default_qpu().submit(bell_pair.to_job())
for sample in result:
    print(f"{sample.state}: {sample.probability}")


## submit a job
@qat.lang.qrout
def circuit(theta):
    qat.lang.RX(theta)(0)
job = circuit.to_job(observable=qat.core.Observable.z(0))
qpu = qat.plugins.ScipyMinimizePlugin() | qat.qpus.get_default_qpu()
result = qpu.submit(job)
result.value #-1
result.parameter_map #pi



# import numpy as np
# from qat.core import Observable as Obs
# from qat.lang import RY, CNOT, qfunc

@qat.lang.qfunc(thetas=2)
def energy(thetas):
    qat.lang.RY(thetas[0])(0)
    qat.lang.RY(thetas[1])(1)
    qat.lang.CNOT(0, 1)
    pauli = qat.core.Observable
    ret = pauli.sigma_z(0) * pauli.sigma_z(1) + pauli.sigma_x(0) * pauli.sigma_x(1) + pauli.sigma_y(0) * pauli.sigma_y(1)
    return ret

hf0 = lambda x: energy(x)
theta_optim = scipy.optimize.minimize(hf0, x0=[-1, 3], jac='2-point', method='L-BFGS-B')
ret0 = scipy.optimize.minimize(energy, x0=[1,1.8], method='L-BFGS-B', options={'disp':True}) #x0=np.array([1.8, 2.8])
ret0.fun #-0.3099330343247272
# Equivalently, one can delegate the minimization to the default qpu which is equiped with a variational optimizer
ret1 = energy.run()
ret1.parameter_map
ret1.value #-3
energy(np.array([-np.pi/2, np.pi]))
print(f"Minimum VQE energy = {ret1.value}")
