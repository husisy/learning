import numpy as np
import stim


sim = stim.TableauSimulator()
sim.set_state_from_stabilizers([
    stim.PauliString("+XZZX_"),
    stim.PauliString("+_XZZX"),
    stim.PauliString("+X_XZZ"),
    stim.PauliString("+ZZX_X"),
    stim.PauliString("+ZZZZZ"),
])
tableau = sim.current_inverse_tableau()**-1
stab_list = tableau.to_stabilizers() #(list,stim.PauliString)
stab_list = [tableau.z_output(i) for i in range(len(tableau))]
# +XZZX_  +_XZZX  +X_XZZ  +ZZX_X  +ZZZZZ
destab_list = [tableau.x_output(i) for i in range(len(tableau))]
# +___Z_  +_Z___  +Z__Z_  +_Z__Z  -_ZXZ_


sim = stim.TableauSimulator()
sim.set_state_from_stabilizers([
    stim.PauliString("+XZZX_"),
    stim.PauliString("+_XZZX"),
    stim.PauliString("+X_XZZ"),
    stim.PauliString("+ZZX_X"),
    stim.PauliString("+ZZZZZ"),
])
tableau = sim.current_inverse_tableau()**-1
circ = tableau.to_circuit(method="elimination")


sim = stim.FlipSimulator(batch_size=2**10)
sim.do(stim.Circuit("M(0.1) 0 1"))
sim.num_qubits #1
x0 = sim.get_measurement_flips() #(np,bool,(2,1024))
x0.mean(axis=1) #around 0.1



tmp0 = '''
X_ERROR(1) 0 1 3
REPEAT 5 {
    H 0
    C_XYZ 1
}
'''
circ = stim.Circuit(tmp0)
circ[0] #X_ERROR(1) 0 1 3
circ[1] #REPEAT 5 {...}

sim = stim.FlipSimulator(batch_size=1, disable_stabilizer_randomization=True)
sim.do(circ)
sim.peek_pauli_flips()
[stim.PauliString("+ZZ_X")]

sim.do(circ[0])
sim.peek_pauli_flips()
[stim.PauliString("+YY__")]

sim.do(circ[1])
sim.peek_pauli_flips()
[stim.PauliString("+YX__")]



sim = stim.FlipSimulator(batch_size=1, disable_stabilizer_randomization=True)
sim.do(stim.Circuit('X 0 1 3'))
sim.peek_pauli_flips() #[stim.PauliString("+____")]

sim.clear()
sim.do(stim.Circuit('X_ERROR(1) 0 1 3'))
sim.peek_pauli_flips() #[stim.PauliString("+XX_X")]

sim.do(stim.Circuit('H 0'))
sim.peek_pauli_flips() #[stim.PauliString("+ZX_X")]


x0 = stim.Circuit('CY 0 1').to_tableau() #[XY ZX] [Z_ ZZ]
assert x0(stim.PauliString('XY')) == stim.PauliString('X_') #"X_" out-of-place
# TODO in-place

tab0 = stim.Tableau.random(5)
sim = stim.TableauSimulator()
sim.set_inverse_tableau(tab0**-1)
assert sim.current_inverse_tableau() == tab0**-1
