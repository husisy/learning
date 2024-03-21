import os
import dpdata

# wget https://dp-public.oss-cn-beijing.aliyuncs.com/community/DeePMD-kit-FastLearn.tar
hf_data = lambda *x: os.path.join('data', 'DeePMD-kit-FastLearn', *x)
# 00.data/OUTCAR: vasp result
# 01.train/input.json: deepmd-kit configuration
# data/ : deepmd-kit training/validation data
hf_tbd00 = lambda *x: os.path.join('tbd00', *x)


data = dpdata.LabeledSystem(hf_data('00.data', 'OUTCAR'))
data.to('deepmd/npy', hf_tbd00('deepmd_data'), set_size=200)
# save every 200 frames of data to set.000/set.001/...
# 1 frame only in 00.data/OUTCAR
# type_map.raw:


# 'vasp/xml'
dsys = dpdata.LabeledSystem('vasprun.h2o.md.10.xml', fmt='vasp/xml')
dsys.to('deepmd/npy', 'deepmd_data', set_size = dsys.get_nframes())


dsys = dpdata.LabeledSystem('OUTCAR', fmt='vasp/outcar')
dsys.to('deepmd/npy', 'deepmd_data', set_size = dsys.get_nframes())
