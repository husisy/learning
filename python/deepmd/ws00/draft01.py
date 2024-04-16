# https://tutorials.deepmodeling.com/en/latest/Tutorials/DeePMD-kit/learnDoc/Handson-Tutorial%28v2.0.3%29.html
import os
import numpy as np

import dpdata

np_rng = np.random.default_rng()

# wget https://dp-public.oss-cn-beijing.aliyuncs.com/community/CH4.tar
hf_data = lambda *x: os.path.join('data', *x)


data = dpdata.LabeledSystem(hf_data('validation_data'), fmt='deepmd/npy')
pred_data = data.predict(dp=hf_data('01.train', 'graph-compress.pb'))


# /Users/zhangc/micromamba/envs/metal/lib/python3.11/site-packages/dpdata/system.py
# driver = dpdata.system.Driver.get_driver('dp')(hf_data('01.train', 'graph-compress.pb'))
# z0 = dpdata.system.LabeledSystem(data=driver.label(data.data.copy()))


# dpdata.driver.Driver.__DriverPlugin
# /Users/zhangc/micromamba/envs/metal/lib/python3.11/site-packages/dpdata/driver.py
# z1 = dpdata.plugin.Plugin()
# z1.register('dp')

import dpdata.plugins.deepmd
driver = dpdata.plugins.deepmd.DPDriver(hf_data('01.train', 'graph-compress.pb'))

# z3 = driver.label(data.data.copy())
ori_sys = dpdata.System.from_dict({"data": data.data.copy()})
ori_sys_copy = ori_sys.copy()
ori_sys.sort_atom_names(type_map=driver.dp.get_type_map())
atype = ori_sys["atom_types"]
ori_sys = ori_sys_copy
coord = ori_sys.data["coords"].reshape((ori_sys.get_nframes(), ori_sys.get_natoms()*3))
if not ori_sys.nopbc:
    cell = ori_sys.data["cells"].reshape((ori_sys.get_nframes(), 9))
else:
    cell = None
e, f, v = driver.dp.eval(coord, cell, atype)
z3 = ori_sys.data.copy()
z3["energies"] = e.reshape((ori_sys.get_nframes(),))
z3["forces"] = f.reshape((ori_sys.get_nframes(), ori_sys.get_natoms(), 3))
z3["virials"] = v.reshape((ori_sys.get_nframes(), 3, 3))

print(np.abs(pred_data['energies']-z3['energies']).max())

driver.dp.graph

# import tensorboard
# import tensorflow as tf
# file_writer = tf.compat.v1.summary.FileWriter('logs', driver.dp.graph)
