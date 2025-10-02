import os
import json
import numpy as np
import matplotlib.pyplot as plt
# plt.ion()

import mqt.qecc
import mqt.qecc.cc_decoder.decoder
import mqt.qecc.circuit_synthesis

# steane code
lattice = mqt.qecc.codes.HexagonalColorCode(distance=3) #side_length=distance
prob = mqt.qecc.cc_decoder.decoder.LightsOut(lattice.faces_to_qubits, lattice.qubits_to_faces)
prob.preconstruct_z3_instance()
syndrome = [True, False, False]
rec, constr_time, solve_time = prob.solve(syndrome)


d = 7  # distance of the triangular code to simulate
p = 0.01  # (bit-flip) error rate
n = 1000  # number of simulations to run
logdir = 'tbd00'
mqt.qecc.cc_decoder.decoder.run("hexagon", distance=d, error_rate=p, nr_sims=n, results_dir=logdir)
tmp0 = os.path.join(logdir, [x for x in os.listdir(logdir) if x.endswith('.json')][0])
with open(tmp0, 'r') as f:
    data = json.load(f)


code = mqt.qecc.CSSCode.from_code_name("Steane")
code.stabs_as_pauli_strings()

non_ft_sp = mqt.qecc.circuit_synthesis.gate_optimal_prep_circuit(code, zero_state=True, max_timeout=2) #8 CNOT
non_ft_sp.circ.draw(output="mpl", initial_state=True)


# https://www.nature.com/articles/srep19578
ft_sp = mqt.qecc.circuit_synthesis.gate_optimal_verification_circuit(non_ft_sp) #NP-complete
ft_sp.draw(output="mpl", initial_state=True)
