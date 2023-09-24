import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import time
import numpy as np
# TODO limit threads

np_rng = np.random.default_rng()

num_repeat = 10
N0_list = [64,128,256,512,1024,2048]
time_list = []

for N0 in N0_list:
    print(N0)
    np0_list = np_rng.uniform(-1, 1, size=(num_repeat,N0,N0))
    for ind0 in range(num_repeat):
        t0 = time.time()
        _ = np.linalg.svd(np0_list[ind0])
        time_list.append(time.time() - t0)
time_list = np.array(time_list).reshape(-1, num_repeat)
np.save('tbd_openblas.npy', time_list)

'''
micromamba create -y -n metal-acc
micromamba install -y -n metal-acc "libblas=*=*accelerate" pip
micromamba activate metal-acc
pip install numpy==1.26

micromamba create -y -n metal-openblas
micromamba install -y -n metal-openblas "libblas=*=*openblas" pip
micromamba activate metal-openblas
pip install numpy==1.26

import numpy as np
time_acc = np.load("tbd_acc.npy")
time_openblas = np.load("tbd_openblas.npy")
'''
