import os
import shutil
import random

def next_tbd_dir(dir0='tbd00', maximum_int=100000):
    if not os.path.exists(dir0): os.makedirs(dir0)
    tmp1 = [x for x in os.listdir(dir0) if x[:3]=='tbd']
    exist_set = {x[3:] for x in tmp1}
    while True:
        tmp1 = str(random.randint(1,maximum_int))
        if tmp1 not in exist_set: break
    tbd_dir = os.path.join(dir0, 'tbd'+tmp1)
    os.mkdir(tbd_dir)
    return tbd_dir


def detect_executable(cmd):
    path = shutil.which(cmd)
    ret = False
    if path is not None:
        ret = os.access(path, os.X_OK)
    return ret

def detect_target_device():
    is_gpu = detect_executable('nvidia-smi')
    is_npu = detect_executable('npu-smi')
    assert not (is_gpu and is_npu)
    tmp0 = {(True,False):'GPU', (False,True):'Ascend', (False,False):'CPU'}
    device = tmp0[(is_gpu,is_npu)]
    return device
