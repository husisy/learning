# https://tutorials.deepmodeling.com/en/latest/Tutorials/DeePMD-kit/learnDoc/Handson-Tutorial%28v2.0.3%29.html
import os
import numpy as np

import dpdata

np_rng = np.random.default_rng()

# wget https://dp-public.oss-cn-beijing.aliyuncs.com/community/CH4.tar
hf_data = lambda *x: os.path.join('data', *x)
# data/00.data/OUTCAR: vasp result
# data/01.train/input.json: deepmd-kit configuration
# data/02.Imp: contains the LAMMPS example script for molecular dynamics simulation


data = dpdata.LabeledSystem(hf_data('00.data', 'OUTCAR'), fmt='vasp/outcar')
len(data) #200
data.get_nframes() #200




# random choose 40 index for validation_data
index_validation = np.sort(np_rng.choice(len(data), size=40, replace=False))
index_training = np.array(sorted(set(range(len(data)))-set(index_validation.tolist())))
data.sub_system(index_training).to('deepmd/npy', hf_data('training_data'))
data.sub_system(index_validation).to('deepmd/npy', hf_data('validation_data'))

'''
cd data/01.train
dp train input.json
dp freeze -o graph.pb
dp compress -i graph.pb -o graph-compress.pb
dp test -m graph-compress.pb -s ../validation_data -n 40 -d results
cd ../..
'''

data = dpdata.LabeledSystem(hf_data('validation_data'), fmt='deepmd/npy')
# data = dpdata.LabeledSystem(hf_data('00.data', 'OUTCAR'), fmt='vasp/outcar')
pred_data = data.predict(dp=hf_data('01.train', 'graph-compress.pb'))

np.sqrt(np.mean((pred_data['energies'] - data['energies'])**2))

# energies(np,float64,200)
z0 = np.loadtxt(hf_data('01.train', 'results.e.out'))
assert np.abs(pred_data['energies']-z0[:,1]).max() < 1e-7

# forces(np,float64,(40,5,3))
z0 = np.loadtxt(hf_data('01.train', 'results.f.out')) #(np,float64,(200,6))
assert np.abs(z0[:,3:].reshape(-1,5,3) - pred_data['forces']).max() < 1e-7

# virials(np,float64,(40,3,3)
z0 = np.loadtxt(hf_data('01.train', 'results.v.out')) #(np,float64,(40,18))
assert np.abs(z0[:,9:].reshape(-1,3,3) - pred_data['virials']).max() < 1e-7

z0 = np.loadtxt(hf_data('01.train', 'results.v_peratom.out')) #(np,float64,(40,18))
# pred_data['atom_pref'].shape

# dp train
import deepmd_utils.main
import deepmd.entrypoints.main
# deepmd.entrypoints.main.main(deepmd_utils.main.parse_args())


