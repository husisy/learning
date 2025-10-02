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


circ = stim.Circuit('''
X_ERROR(1) 0
CX 0 1
''')
sim = stim.FlipSimulator(batch_size=1, disable_stabilizer_randomization=True) #TODO
sim.do(circ)
x0 = sim.peek_pauli_flips()[0] #XX



tmp0 = '''
RZ 5 6 7 8
H 5 6 7 8
CX 5 0
CZ 5 1
CZ 5 2
CX 5 3

CX 6 1
CZ 6 2
CZ 6 3
CX 6 4

CX 7 2
CZ 7 3
CZ 7 4
CX 7 0

CX 8 3
CZ 8 4
CZ 8 0
CX 8 1
H 5 6 7 8
'''
circ_syndrome = stim.Circuit(tmp0)
sim = stim.FlipSimulator(batch_size=1, disable_stabilizer_randomization=True) #TODO
sim.do(stim.Circuit(tmp0))
result = sim.peek_pauli_flips()

tmp0 = [(i,s) for i in range(5) for s in 'XYZ']
for ind0, s in tmp0:
    sim.clear()
    sim.do(stim.Circuit(f'{s}_ERROR(1) {ind0}') + circ_syndrome)
    result = sim.peek_pauli_flips()[0]
    print(f'syndrome({s}{ind0}):', ''.join(['_!'[x] for x in result[5:].to_numpy()[0].tolist()]))
