import numpy as np

import qecc

code = qecc.StabilizerCode.perfect_5q_code()
print(next(code.encoding_cliffords()).circuit_decomposition())

qecc.I, qecc.X, qecc.Y, qecc.Z
qecc.Pauli('I'), qecc.Pauli('X'), qecc.Pauli('Y'), qecc.Pauli('Z')
qecc.Pauli('XZZXI')
qecc.Pauli('XYZ')
qecc.Pauli('XYZ', phase=2)

qecc.Pauli('X') * qecc.Pauli('Y') #multiplication
qecc.Pauli('X') & qecc.Pauli('Y') #tensor product
- qecc.Pauli('X') #negation


z0 = qecc.Pauli('XZZXI')
len(z0), z0.nq #num_qubit

qecc.Pauli('XYZXYZ').permute_op('ZXY')

qecc.Pauli('XYZXYZ').as_bsv()

tmp0 = qecc.Clifford([qecc.Pauli('XI',2),qecc.Pauli('IX')], map(qecc.Pauli,['ZI','IZ']))
qecc.Pauli.from_clifford(tmp0)


# Clifford
qecc.Clifford(['IZ','XZ'],['XI',qecc.Unspecified])
z0 = qecc.Clifford(['XX', 'IX'], ['ZI', 'ZZ'])
z0(qecc.X & qecc.Y)

qecc.Clifford(['XX', 'IX'], ['ZI', 'ZZ']) * qecc.Clifford(['XI', 'IZ'], ['ZI', 'IX'])

# paulilist
qecc.PauliList(['I', 'X', 'Y', 'Z'])
qecc.PauliList('XYZ', 'YZX', 'ZXY')
qecc.PauliList(['I', 'X', 'Y', 'Z']) & qecc.X

qecc.Clifford(qecc.PauliList('XX', qecc.Unspecified), qecc.PauliList(qecc.Unspecified, qecc.Pauli('ZZ', phase=2)))

qecc.PauliList('XXX', 'YIY', 'ZZI').pad(extra_bits=2, lower_right=qecc.PauliList('IX','ZI'))

qecc.PauliList('XX', qecc.Unspecified, qecc.Unspecified, 'ZZ').stabilizer_subspace()

qecc.PauliList('XXI', 'IXX').stabilizer_subspace() #codewords of the phase-flip code

## binary symplectic vector
x0 = [1, 0, 1]
x1 = [0, 1, 1]
qecc.BinarySymplecticVector(x0,x1)
qecc.BinarySymplecticVector(x0+x1)
z0 = qecc.Pauli('XYIYIIZ',2).as_bsv()
z0.x
z0.z
z0.as_pauli()

z0 = qecc.BinarySymplecticVector([1,0,1],[0,1,1])
z1 = qecc.Pauli('YYZ').as_bsv()
z0.bsip(z1)

list(qecc.all_pauli_bsvs(1))

list(qecc.constrained_set(map(lambda s: qecc.Pauli(s).as_bsv(), ['XY','ZZ']),[1,0]))

qecc.constrained_set(map(lambda s: qecc.Pauli(s).as_bsv(), ['XY','ZZ']), [1,0])

hf0 = lambda s: qecc.Pauli(s).as_bsv()
list(qecc.constrained_set(list(map(hf0, ['XY','ZZ'])), [1,0]))


## stabilizer code
qecc.StabilizerCode(['ZZI', 'IZZ'], ['XXX'], ['ZZZ'])
qecc.StabilizerCode.bit_flip_code(1)
qecc.StabilizerCode.phase_flip_code(1)
z0 = qecc.StabilizerCode.perfect_5q_code()
z0.nq
z0.nq_logical
z0.distance
next(z0.encoding_cliffords())
z0 & z0
qecc.StabilizerCode.bit_flip_code(1).concatenate(qecc.StabilizerCode.phase_flip_code(1))


## circuit manipulation and simulation
loc = qecc.Location('CNOT', 0, 2)
loc.as_clifford()
loc.nq
circ = qecc.Circuit(('CNOT', 0, 2), ('H', 1), ('X', 0))
circ.nq
circ.depth
circ.size
len(circ)

circ = qecc.Circuit(('CNOT', 0, 2), ('H', 1), ('X', 0), ('H', 1))
circ.cancel_selfinv_gates()

circ = qecc.Circuit(('CZ', 0, 2), ('H', 1), ('X', 0))
circ.replace_cz_by_cnot()
list(circ.group_by_time())

circ = qecc.Circuit(('CZ', 0, 2), ('H', 1), ('X', 0))
circ.as_clifford()


## constraint solver
list(qecc.solve_commutation_constraints(qecc.PauliList('XXI', 'IZZ', 'IYI'), qecc.PauliList('YIY')))
tmp0 = qecc.solve_commutation_constraints(qecc.PauliList('XXI', 'IZZ', 'IYI'), qecc.PauliList('YIY'))
[x for x in tmp0 if (x.wt<=2)]
