import os
import random
import numpy as np

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hfe_r5 = lambda x,y,eps=1e-5: round(hfe(x,y,eps),5)

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

def hf_softmax(data, axis=-1):
    tmp0 = data - data.max(axis=axis, keepdims=True)
    tmp1 = np.exp(tmp0)
    return tmp1 / tmp1.sum(axis=axis, keepdims=True)

hf_logistic = lambda x: 1 / (1 + np.exp(-x))
