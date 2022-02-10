import os
import h5py
import numpy as np

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

# mode r/w/a
np0 = np.random.randint(233, size=(100,))
with h5py.File(hf_file('tbd00.hdf5'), 'w', libver='latest') as fid:
    group0 = fid.create_group('group0')
    dataset0 = fid.create_dataset('mydataset', (100,), dtype='i')
    dataset0.attrs['my-info'] = 'np0'
    dataset0[:] = np0
    group10 = fid.create_group('group1/subgroup10')

# with h5py.File(hf_file('tbd00.hdf5'), 'r') as fid:
fid = h5py.File(hf_file('tbd00.hdf5'), 'r')
fid.keys()
fid.name
h5_dataset0 = fid['mydataset']
h5_dataset0.name
h5_dataset0.shape
h5_dataset0.dtype #(np,int32)
h5_dataset0.attrs['my-info']
np1 = h5_dataset0[:]
assert np.all(np0==np1)
# fid.visit()
fid.close()

# dtype=str
z0 = ['2', '23', '233']
with h5py.File(hf_file('tbd00.hdf5'), 'w') as fid:
    dt = h5py.special_dtype(vlen=str)
    dataset0 = fid.create_dataset('mydataset', shape=(len(z0),), dtype=dt)
    for ind0,x in enumerate(z0):
        dataset0[ind0] = x
