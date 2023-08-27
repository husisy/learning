import numpy as np

import tequila


tequila.show_available_simulators()
# backend         | wfn        | sampling   | noise      | installed
# --------------------------------------------------------------------
# qulacs_gpu      | False      | False      | False      | False
# qulacs          | True       | True       | True       | True
# qibo            | False      | False      | False      | False
# qiskit          | False      | False      | False      | False
# cirq            | True       | True       | True       | True
# pyquil          | False      | False      | False      | False
# symbolic        | True       | False      | False      | True
# qlm             | False      | False      | False      | False

tequila.show_available_optimizers()
# available methods for optimizer modules found on your system:
# method               | optimizer module
# --------------------------
# NELDER-MEAD          | scipy
# COBYLA               | scipy
# POWELL               | scipy
# SLSQP                | scipy
# L-BFGS-B             | scipy
# BFGS                 | scipy
# CG                   | scipy
# TNC                  | scipy
# TRUST-KRYLOV         | scipy
# NEWTON-CG            | scipy
# DOGLEG               | scipy
# TRUST-NCG            | scipy
# TRUST-EXACT          | scipy
# TRUST-CONSTR         | scipy
# adam                 | gd
# adagrad              | gd
# adamax               | gd
# nadam                | gd
# sgd                  | gd
# momentum             | gd
# nesterov             | gd
# rmsprop              | gd
# rmsprop-nesterov     | gd
# spsa                 | gd
# Supported optimizer modules:  ['scipy', 'phoenics', 'gpyopt', 'gd']
# Installed optimizer modules:  ['scipy', 'gd']

circuit = (
    tequila.gates.H(target=0) + tequila.gates.CNOT(target=1,control=0)
    + tequila.gates.Ry(angle=1.0, target=0) + tequila.gates.Y(power=0.5, target=0)
    + tequila.gates.Trotterized(generators=[tequila.paulis.X(0)*tequila.paulis.Y(1)], angles=[1.0], steps=1)
    + tequila.gates.Rp(angle=1.0, paulistring="X(0)Y(1)")
)
# tequila.draw(circuit)
# 0: ───H───@───Y^0.318───Y^0.5───Y^-0.5───@─────────────@───Y^0.5────Y^-0.5───@─────────────@───Y^0.5────
#           │                              │             │                     │             │
# 1: ───────X───X^0.5──────────────────────X───Z^0.318───X───X^-0.5───X^0.5────X───Z^0.318───X───X^-0.5───
state = tequila.simulate(circuit)
measurement = tequila.simulate(circuit, samples=10)
measurement('10') #measurement(2)
measurement = tequila.compile(circuit, samples=10)(samples=10, read_out_qubits=[1])

circuit = tequila.gates.Ry(angle="a", target=0)
state = tequila.simulate(circuit, variables={"a" : np.pi**2})
circuit.extract_variables()


var_a = tequila.Variable("a")
circuit = tequila.gates.Ry(angle=(var_a*np.pi)**2, target=0)
state = tequila.simulate(circuit, variables={"a" : 1.0})



hf0 = lambda x: tequila.numpy.exp(x**2)
a = tequila.Variable("a")
tmp0 = (a*np.pi).apply(hf0) # exp((a*pi)**2)
circuit = tequila.gates.Ry(angle=tmp0, target=0)
state = tequila.simulate(circuit, variables={"a" : 1.0})


H = tequila.paulis.X([0,1]) + 0.5*tequila.paulis.Z([0,1])
U = tequila.gates.Ry("a", 0) + tequila.gates.X(1,control=0)
E = tequila.ExpectationValue(H=H, U=U)
hf0 = tequila.compile(E)
hf0({'a':1.0}) #1.3414709848078965

gradient_a = tequila.grad(E, 'a')
hf_grad = tequila.compile(gradient_a)
hf_grad({'a':1}) #0.5403022766113281
