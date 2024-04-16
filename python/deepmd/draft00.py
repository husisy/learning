import os
import numpy as np

import dpdata

# wget https://dp-public.oss-cn-beijing.aliyuncs.com/community/DeePMD-kit-FastLearn.tar
hf_data = lambda *x: os.path.join('data', 'DeePMD-kit-FastLearn', *x)
# 00.data/OUTCAR: vasp result
# 01.train/input.json: deepmd-kit configuration
# data/ : deepmd-kit training/validation data
hf_tbd00 = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_tbd00()):
    os.makedirs(hf_tbd00())


data = dpdata.LabeledSystem(hf_data('00.data', 'OUTCAR'))
data.get_nframes() #1
datadir = hf_tbd00('deepmd_data')
data.to('deepmd/npy', datadir, set_size=200)
# save every 200 frames of data to set.000/set.001/...
# 1 frame only in 00.data/OUTCAR
# type_map.raw:
x0 = np.load(hf_tbd00('deepmd_data', 'set.000', 'box.npy')) #(np,float64,(1,9))
x0 = np.load(hf_tbd00('deepmd_data', 'set.000', 'coord.npy'))  #(np,float64,(1,18))
x0 = np.load(hf_tbd00('deepmd_data', 'set.000', 'energy.npy'))  #(np,float64,(1,))
x0 = np.load(hf_tbd00('deepmd_data', 'set.000', 'force.npy'))  #(np,float64,(1,18))
x0 = np.load(hf_tbd00('deepmd_data', 'set.000', 'virial.npy'))  #(np,float64,(1,9))
