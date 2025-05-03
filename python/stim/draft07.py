# https://software.roffe.eu/ldpc/classical_coding.html
# import ldpc.codes.hamming_code
import numpy as np

import ldpc
import ldpc.codes
import ldpc.mod2
import ldpc.code_util
import ldpc.monte_carlo_simulation
import bposd
import bposd.css
import bposd.hgp

np_rng = np.random.default_rng()

H = ldpc.codes.rep_code(distance=5) #repetition code, parity check matrix (scipy.sparse.csr,uint8,(4,5))
H = ldpc.codes.hamming_code(rank=3) #(scipy.sparse.csr,uint8,(3,7))
n = H.shape[1] #block length of the code
k = n-ldpc.mod2.rank(H) #the dimension of the code computed using the rank-nullity theorem.
d = ldpc.code_util.compute_code_distance(H) #code distance, should only be used for small codes
print(f"Hamming code parameters: [n={n},k={k},d={d}]")


# ldpc code
H = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
       [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
       [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0]], dtype=np.uint8) #(np,uint8,(9,14))
n, k, d_estimate = ldpc.code_util.compute_code_parameters(H) #[14,5,5]
G = ldpc.code_util.construct_generator_matrix(H) #(scipy.sparse.csr,uint8,(5,14))
assert np.all((G @ H.T) % 2==0)

error_rate = 0.1
decoder = ldpc.BpDecoder(H, error_rate=error_rate, max_iter=n, bp_method="product_sum")
num_shot = 10000
error = np_rng.choice(2, size=(num_shot, n), p=[1-error_rate, error_rate]).astype(np.uint8)
syndrome = (error @ H.T) % 2
num_fail = sum(np.any(((decoder.decode(x)+y)%2)!=0) for x,y in zip(syndrome,error))
std_err = np.sqrt(num_fail*(num_shot-num_fail)/num_shot**3)
print(f"logical error rate: {num_fail/num_shot} ± {std_err:.3}") #0.1387 ± 0.00346
mc_sim = ldpc.monte_carlo_simulation.MonteCarloBscSimulation(H, error_rate=error_rate, target_run_count=10000, Decoder=decoder, tqdm_disable=True)
result = mc_sim.run()
print(f"logical error rate: {result['logical_error_rate']} ± {result['logical_error_rate_eb']:.3}") #0.1324 ± 0.00339



H = ldpc.codes.rep_code(distance=3) #(scipy.sparse.csr,uint8,(2,3))
n = H.shape[1] #the codeword length
bpd = ldpc.BpDecoder(
    H,
    error_rate=0.1, # the error rate on each bit
    max_iter=n, #the maximum iteration depth for BP
    bp_method="product_sum", #BP method. The other option is `minimum_sum'
)
error = np.array([0,1,0], dtype=np.uint8)
syndrome = (H@error) % 2 #(1,1)
recovery = bpd.decode(syndrome) #(0,1,0)
bpd1 = ldpc.BpDecoder(H, max_iter=n, bp_method="product_sum", channel_probs=[0.1,0,0.1])
# channel probability probabilities. Will overide error rate.
recovery1 = bpd1.decode(syndrome) #(1,0,1)
