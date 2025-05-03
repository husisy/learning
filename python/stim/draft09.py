import numpy as np

import ldpc
import ldpc.codes
import bposd
import bposd.css
import bposd.hgp

## bposd
h = ldpc.codes.hamming_code(rank=3)
steane_code = bposd.css.css_code(hx=h, hz=h) #create Steane code where both hx and hz are Hamming codes
steane_code.test() #True
steane_code.hx
steane_code.hz
steane_code.lx #logical x
steane_code.lz #logical z

code = steane_code
# bposd decoder
# ldpc.BpOsdDecoder #TODO
bpd = ldpc.bposd_decoder(
    code.hz,#the parity check matrix
    error_rate=0.05,
    channel_probs=[None], #assign error_rate to each qubit. This will override "error_rate" input variable
    max_iter=code.N, #the maximum number of iterations for BP)
    bp_method="ms",
    # ms_scaling_factor=0.0, #min sum scaling factor. If set to zero the variable scaling factor method is used
    osd_method="osd_cs", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
    osd_order=7 #the osd search depth
)

error = np.zeros(code.N, dtype=np.uint8)
error[[5]] = 1
syndrome = (code.hz@error) %2
recovery = bpd.decode(syndrome)
residual_error = (recovery+error) %2
#Decoding is successful if the residual error commutes with the logical operators
code.lz@residual_error%2 #0
code.lx@residual_error%2 #0


# TODO surface code, pymatching code-capacity threshold
