import numpy as np

import ldpc
import ldpc.codes
import ldpc.mod2
import ldpc.code_util
import bposd
import bposd.css
import bposd.hgp

H = ldpc.codes.rep_code(distance=5) #the repetition code parity check matrix, (np,int64,(4,5))
H = ldpc.codes.hamming_code(rank=3) #(np,int64,(3,7))
n = H.shape[1] #block length of the code
k = n-ldpc.mod2.rank(H) #the dimension of the code computed using the rank-nullity theorem.
d = ldpc.code_util.compute_code_distance(H) #code distance, should only be used for small codes
print(f"Hamming code parameters: [n={n},k={k},d={d}]")



H = ldpc.codes.rep_code(distance=3) #(np,int64,(2,3))
n = H.shape[1] #the codeword length
bpd = ldpc.bp_decoder(
    H,
    error_rate=0.1, # the error rate on each bit
    max_iter=n, #the maximum iteration depth for BP
    bp_method="product_sum", #BP method. The other option is `minimum_sum'
    channel_probs=[None] #channel probability probabilities. Will overide error rate.
)
x = bpd.decode(np.array([0,1,1], dtype=np.uint8)) #(1,1,1)
error = np.array([0,1,0])
syndrome = H@error%2 #(1,1)
decoding = bpd.decode(syndrome) #(0,1,0)


H = ldpc.codes.rep_code(distance=3) #(np,int64,(2,3))
n = H.shape[1] #the codeword length
bpd=ldpc.bp_decoder(
    H,
    max_iter=n,
    bp_method="product_sum",
    channel_probs=[0.1,0,0.1] #channel probability probabilities. Will overide error rate.
)
error=np.array([1,0,1])
syndrome=H@error%2 #(1,1)
decoding=bpd.decode(syndrome) #(1,0,1)


## bposd
h = ldpc.codes.hamming_code(rank=3)
steane_code = bposd.css.css_code(hx=h, hz=h) #create Steane code where both hx and hz are Hamming codes
steane_code.test() #True
steane_code.hx
steane_code.hz
steane_code.lx #logical x
steane_code.lz #logical z

# not all (hx,hz) pairs are valid codes
x = ldpc.codes.rep_code(7)
bposd.css.css_code(x,x).test() #False

# hypergraph product
h = ldpc.codes.rep_code(3)
surface_code = bposd.hgp.hgp(h1=h, h2=h, compute_distance=True) #set compute_distance=False for larger codes
surface_code.test() #True

# bposd decoder
bpd = ldpc.bposd_decoder(
    surface_code.hz,#the parity check matrix
    error_rate=0.05,
    channel_probs=[None], #assign error_rate to each qubit. This will override "error_rate" input variable
    max_iter=surface_code.N, #the maximum number of iterations for BP)
    bp_method="ms",
    ms_scaling_factor=0, #min sum scaling factor. If set to zero the variable scaling factor method is used
    osd_method="osd_cs", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
    osd_order=7 #the osd search depth
)

error = np.zeros(surface_code.N, dtype=np.uint8)
error[[5,12]] = 1
syndrome = (surface_code.hz@error) %2
bpd.decode(syndrome)
bpd.osdw_decoding
residual_error = (bpd.osdw_decoding+error) %2
#Decoding is successful if the residual error commutes with the logical operators
surface_code.lz@residual_error%2 #0
surface_code.lx@residual_error%2 #0
