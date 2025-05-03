# https://pymatching.readthedocs.io/en/stable/
import numpy as np
import matplotlib.pyplot as plt
import stim
import pymatching
import scipy.sparse

np_rng = np.random.default_rng()


circ = stim.Circuit.generated("surface_code:rotated_memory_x", distance=5, rounds=5, after_clifford_depolarization=0.005)
# circ.to_file('tbd00.txt')
# circ = stim.Circuit.from_file('tbd00.txt')
matcher = pymatching.Matching.from_detector_error_model(circ.detector_error_model(decompose_errors=True))
sampler = circ.compile_detector_sampler()
syndrome, actual_observables = sampler.sample(shots=1000, separate_observables=True)
# syndrome (np,bool,(1000,120))
# actual_observables (np,bool,(1000,1))
predicted_observables = np.stack([matcher.decode(x) for x in syndrome]) #(np,uint8,(1000,1))
predicted_observables = matcher.decode_batch(syndrome)
num_errors = (predicted_observables[:,0]!=actual_observables[:,0]).sum()


H = scipy.sparse.csc_matrix([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]])
weights = np.array([4, 3, 2, 3, 4])   # Set arbitrary weights for illustration
matcher = pymatching.Matching(H, weights=weights)
prediction = matcher.decode(np.array([0, 1, 0, 1])) #[0,0,1,1,0]
# prediction, solution_weight = matcher.decode(np.array([0, 1, 0, 1]), return_weight=True) #solution_weight=2+3=5


## raise error, pymatching only support Tanner graph with degree 2 on check nodes
# H = scipy.sparse.csc_matrix([[1, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]])
# weights = np.array([4, 3, 2, 3, 4])
# matcher = pymatching.Matching.from_check_matrix(H, weights=weights)


H = scipy.sparse.csc_matrix([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]])
observables = scipy.sparse.csc_matrix([[1, 1, 1, 1, 1]])
noise = 0.1
weights = np.ones(H.shape[1]) * np.log((1-noise)/noise)
matcher = pymatching.Matching.from_check_matrix(H, weights=weights)
matcher1 = pymatching.Matching.from_check_matrix(H, weights=weights, faults_matrix=observables) #equivalent
num_shots = 1000
num_fail = 0
for i in range(num_shots):
    error = (np.random.random(H.shape[1]) < noise).astype(np.uint8)
    syndrome = H@error % 2
    # prediction = matcher.decode(syndrome)
    # prediction_observables = (observables@prediction) % 2
    prediction_observables = matcher1.decode(syndrome)
    if (prediction_observables) != ((observables@error)%2):
        num_fail = num_fail + 1


distance_list = [7,9,11,13]
noise_list = np.linspace(0.05, 0.6, 15)
num_shot = int(1e5)
num_fail_list = []
for d in distance_list:
    for noise in noise_list:
        print(d, noise)
        H = np.eye(d-1, d, dtype=np.uint8) + np.diag(np.ones(d-1, dtype=np.uint8), k=1)[:-1]
        # observables = scipy.sparse.csc_matrix([[1]+[0]*(d-1)])
        observables = scipy.sparse.csc_matrix([[1]*d])
        weights = np.ones(H.shape[1]) * np.log((1-noise)/noise)
        matcher = pymatching.Matching.from_check_matrix(H, weights=weights, faults_matrix=observables)
        noise = (np_rng.uniform(0,1 ,size=(num_shot,H.shape[1])) < noise).astype(np.uint8)
        syndrome = (noise @ H.T) % 2
        actual_observables = (noise @ observables.T) % 2
        predicted_observables = matcher.decode_batch(syndrome)
        num_fail_list.append(np.sum(np.any(predicted_observables != actual_observables, axis=1)))
num_fail_list = np.array(num_fail_list).reshape(len(distance_list), len(noise_list))
fig,ax = plt.subplots()
ax.plot(noise_list, noise_list, label='y=x')
for ind0 in range(len(distance_list)):
    ax.plot(noise_list, num_fail_list[ind0]/num_shot, label=f'd={distance_list[ind0]}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)
