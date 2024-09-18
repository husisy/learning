# https://github.com/quantumgizmos/bp_osd/tree/main/examples
import numpy as np

import bposd.hgp
import bposd.css_decode_sim

tmp0 = '''
1 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0
0 0 1 0 0 0 1 1 0 0 0 0 1 0 0 0
0 0 0 1 1 0 1 0 0 0 0 0 0 1 0 0
0 0 0 0 0 1 0 0 1 1 0 0 0 0 0 1
0 1 0 0 0 0 0 1 1 0 0 1 0 0 0 0
0 0 0 0 0 0 0 0 1 0 0 0 1 1 1 0
1 0 0 0 0 0 0 1 0 0 0 0 0 1 0 1
0 0 0 1 0 1 0 0 0 0 1 0 1 0 0 0
0 0 1 1 0 0 0 0 0 0 0 1 0 0 0 1
0 0 0 0 1 0 0 0 0 1 1 1 0 0 0 0
0 1 0 0 0 0 1 0 0 0 1 0 0 0 1 0
1 0 1 0 0 0 0 0 0 1 0 0 0 0 1 0
'''
tmp1 = '''
0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1
0 1 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 1
0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 1 1 0 0 0
0 0 1 0 0 0 1 0 0 0 0 0 0 1 0 0 1 0 0 0
0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 1 0
0 0 0 0 1 0 0 0 0 0 1 0 0 0 1 1 0 0 0 0
0 0 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 1 0 0
0 1 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
0 0 0 0 0 1 0 0 1 0 0 0 1 0 1 0 0 0 0 0
0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 0 1 0 0 0
0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 1 0
1 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 1
1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0
1 0 1 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0
0 1 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 1 0 0
'''
tmp2 = '''
0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 0 0 0 0 0
0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0
0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1
0 0 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 1
0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 1 0 0 1 0 0 0 0
0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0
1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 1
0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 1 0
0 1 0 0 1 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 1 0 0 0 1 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0
0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 1 1 0 0
1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0
0 0 1 0 0 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 1 0 0 1 0 0
0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 0 0 0 0 0
0 0 0 1 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 1 0 0 0
'''
tmp3 = [('mkmn_16_4_6',tmp0), ('mkmn_20_5_8',tmp1), ('mkmn_24_6_10',tmp2)]
hf0 = lambda v: np.array([[int(y) for y in x.split(' ')] for x in v.strip().split('\n')], dtype=np.uint8)
classical_seed_codes = {k:hf0(v) for k,v in tmp3}
hgp_code_dict = dict()
for code,seed_code in classical_seed_codes.items():
    qcode = bposd.hgp.hgp(seed_code, compute_distance=True)
    qcode.canonical_logicals()
    qcode.test()
    hgp_code_dict[code] = qcode

qcode = hgp_code_dict['mkmn_16_4_6'] #construct quantum LDPC code using the symmetric hypergraph product
osd_options = dict(error_rate=0.05, target_runs=1000, xyz_error_bias=[0,0,1], bp_method="ms", ms_scaling_factor=0,
            osd_method="OSD_CS", osd_order=42, channel_update=None, seed=42, max_iter=0, output_file="tbd00.json")
lk = bposd.css_decode_sim.css_decode_sim(hx=qcode.hx, hz=qcode.hz, **osd_options)
