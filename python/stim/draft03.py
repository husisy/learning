# https://pymatching.readthedocs.io/en/stable/toric-code-example.html
import numpy as np
import stim
import pymatching
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.sparse
# from scipy.sparse import hstack, kron, eye, csc_matrix, block_diag


np_rng = np.random.default_rng()

def repetition_code(n):
    """
    Parity check matrix of a repetition code with length n.
    """
    row_ind, col_ind = zip(*((i, j) for i in range(n) for j in (i, (i+1)%n)))
    data = np.ones(2*n, dtype=np.uint8)
    ret = scipy.sparse.csc_matrix((data, (row_ind, col_ind)))
    return ret


def toric_code_x_stabilisers(L):
    """
    Sparse check matrix for the X stabilisers of a toric code with
    lattice size L, constructed as the hypergraph product of
    two repetition codes.
    """
    Hr = repetition_code(L)
    tmp0 = scipy.sparse.kron(Hr, scipy.sparse.eye(Hr.shape[1]))
    tmp1 = scipy.sparse.kron(scipy.sparse.eye(Hr.shape[0]), Hr.T)
    H = scipy.sparse.hstack([tmp0, tmp1], dtype=np.uint8)
    ret = H.tocsc()
    return ret


def toric_code_x_logicals(L):
    """
    Sparse binary matrix with each row corresponding to an X logical operator
    of a toric code with lattice size L. Constructed from the
    homology groups of the repetition codes using the Kunneth
    theorem.
    """
    H1 = scipy.sparse.csc_matrix(([1], ([0],[0])), shape=(1,L), dtype=np.uint8)
    H0 = scipy.sparse.csc_matrix(np.ones((1, L), dtype=np.uint8))
    x_logicals = scipy.sparse.block_diag([scipy.sparse.kron(H1, H0), scipy.sparse.kron(H0, H1)])
    ret = x_logicals.tocsc()
    return ret


def demo_code_capacity_threshold():
    num_shots = 5000
    Ls = [4,8,12]
    noise_list = np.linspace(0.01, 0.2, 9)
    num_fail_list = []
    for L in Ls:
        print("Simulating L={}...".format(L))
        checkX = toric_code_x_stabilisers(L)
        logicalX = toric_code_x_logicals(L)
        for p in noise_list:
            matcher = pymatching.Matching.from_check_matrix(checkX, weights=np.log((1-p)/p), faults_matrix=logicalX)
            error = (np.random.random((num_shots, checkX.shape[1])) < p).astype(np.uint8)
            syndrome = (error @ checkX.T) % 2
            actual_observables = (error @ logicalX.T) % 2
            predicted_observables = matcher.decode_batch(syndrome)
            num_fail_list.append(np.sum(np.any(predicted_observables != actual_observables, axis=1)))
    num_fail_list = np.array(num_fail_list).reshape(len(Ls), len(noise_list))

    fig,ax = plt.subplots()
    for ind0 in range(len(Ls)):
        tmp0 = num_fail_list[ind0]/num_shots
        std_err = (tmp0*(1-tmp0)/num_shots)**0.5
        ax.errorbar(noise_list, tmp0, yerr=std_err, label="L={}".format(L))
    ax.set_xlabel("Physical error rate")
    ax.set_ylabel("Logical error rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


def num_decoding_failures_noisy_syndromes(H, logicals, p, q, num_shots, repetitions):
    matching = pymatching.Matching(H, weights=np.log((1-p)/p),
                repetitions=repetitions, timelike_weights=np.log((1-q)/q), faults_matrix=logicals)
    num_stabilisers, num_qubits = H.shape
    num_fail = 0
    for _ in tqdm(range(num_shots)):
        error = (np_rng.uniform(0,1,size=(num_qubits,repetitions)) < p).astype(np.uint8)
        noise_cumulative = (np.cumsum(error, 1) % 2).astype(np.uint8)
        syndrome = H@noise_cumulative % 2
        syndrome_error = (np_rng.uniform(0,1,size=(num_stabilisers,repetitions)) < q).astype(np.uint8)
        syndrome_error[:,-1] = 0 # Perfect measurements in last round to ensure even parity
        noisy_syndrome = (syndrome + syndrome_error) % 2
        # Convert to difference syndrome
        noisy_syndrome[:,1:] = (noisy_syndrome[:,1:] - noisy_syndrome[:,0:-1]) % 2
        predicted_logicals_flipped = matching.decode(noisy_syndrome)
        actual_logicals_flipped = noise_cumulative[:,-1]@logicals.T % 2
        if not np.array_equal(predicted_logicals_flipped, actual_logicals_flipped):
            num_fail += 1
    return num_fail


num_shots = 5000
Ls = [8,10,12]
ps = np.linspace(0.02, 0.04, 7)
num_fail_list = []
for L in Ls:
    print("Simulating L={}...".format(L))
    Hx = toric_code_x_stabilisers(L)
    logX = toric_code_x_logicals(L)
    for p in ps:
        num_fail_list.append(num_decoding_failures_noisy_syndromes(Hx, logX, p, p, num_shots, L))
num_fail_list = np.array(num_fail_list).reshape(len(Ls), len(ps))

fig,ax = plt.subplots()
for ind0 in range(len(Ls)):
    logical_errors = num_fail_list[ind0]/num_shots
    std_err = (logical_errors*(1-logical_errors)/num_shots)**0.5
    ax.errorbar(ps, logical_errors, yerr=std_err, label=f"L={L}")
ax.set_yscale("log")
ax.set_xlabel("Physical error rate")
ax.set_ylabel("Logical error rate")
ax.legend(loc=0)
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)




num_shots = 20000
Ls = [5,9,13]
ps = np.linspace(0.004, 0.01, 7)
num_fail_list = []
for L in Ls:
    print("Simulating L={}...".format(L))
    for p in ps:
        circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance=L, rounds=L,
                    after_clifford_depolarization=p, before_round_data_depolarization=p,
                    after_reset_flip_probability=p, before_measure_flip_probability=p)
        model = circuit.detector_error_model(decompose_errors=True)
        matching = pymatching.Matching.from_detector_error_model(model)
        sampler = circuit.compile_detector_sampler()
        syndrome, actual_observables = sampler.sample(shots=num_shots, separate_observables=True)
        predicted_observables = matching.decode_batch(syndrome)
        num_fail_list.append(np.sum(np.any(predicted_observables != actual_observables, axis=1)))
        num_errors = np.sum(np.any(predicted_observables != actual_observables, axis=1))
num_fail_list = np.array(num_fail_list).reshape(len(Ls), len(ps))


fig,ax = plt.subplots()
for ind0 in range(len(Ls)):
    logical_errors = num_fail_list[ind0]/num_shots
    std_err = (logical_errors*(1-logical_errors)/num_shots)**0.5
    ax.errorbar(ps, logical_errors, yerr=std_err, label="L={}".format(L))
ax.set_yscale("log")
ax.set_xlabel("Physical error rate")
ax.set_ylabel("Logical error rate")
ax.legend(loc=0)
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)


## TODO
L = 5
p = 0.233
circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance=L, rounds=L,
            after_clifford_depolarization=p, before_round_data_depolarization=p,
            after_reset_flip_probability=p, before_measure_flip_probability=p)
circuit.to_file("tbd00.stim")
