import numpy as np
import stim

circ = stim.Circuit()
circ.append("H", [0])
circ.append("CNOT", [0, 1]) #bell state
circ.append("M", [0, 1]) #measure
# |00> + |11>
print(repr(circ))
# stim.Circuit('''
#     H 0
#     CX 0 1
#     M 0 1
# ''')
x0 = circ.compile_sampler().sample(shots=5)
# (np,bool,(5,2))


circ = stim.Circuit()
circ.append("H", [0])
circ.append("CNOT", [0, 1])
circ.append("M", [0, 1])
circ.append('DETECTOR', [stim.target_rec(-1), stim.target_rec(-2)])
x0 = circ.compile_detector_sampler().sample(shots=5)



circ = stim.Circuit()
circ.append("H", [0])
circ.append("CNOT", [0, 1])
circ.append("X_ERROR", [0, 1], 0.2)
circ.append("M", [0, 1])
circ.append('DETECTOR', [stim.target_rec(-1), stim.target_rec(-2)])
x0 = circ.compile_detector_sampler().sample(shots=5) #one error probability: 0.8*0.2*2


circ = stim.Circuit.generated("repetition_code:memory", rounds=30, distance=9,
        before_round_data_depolarization=0.03, before_measure_flip_probability=0.01)
# print(repr(circ))
