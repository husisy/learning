import time
import numpy as np
import cupy as cp
import pandas as pd
import torch
import tensorflow as tf

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

def demo_float32_diff_precision():
    np0 = np.random.randn(128, 784).astype(np.float32)
    np1 = np.random.randn(784, 100).astype(np.float32)
    ret_np = np.matmul(np0, np1)
    ret_ = np.matmul(np0, np1)

    ret_cp = cp.matmul(cp.array(np0), cp.array(np1)).get()

    device_cpu = torch.device('cpu')
    device_gpu = torch.device('cuda')
    ret_torch_cpu = torch.matmul(torch.tensor(np0, device=device_cpu), torch.tensor(np1, device=device_cpu)).numpy().copy()
    ret_torch_gpu = torch.matmul(torch.tensor(np0, device=device_gpu), torch.tensor(np1, device=device_gpu)).cpu().numpy()

    with tf.device('/CPU:0'):
        ret_tf_cpu = tf.linalg.matmul(tf.constant(np0), tf.constant(np1)).numpy()
    with tf.device('/GPU:0'):
        ret_tf_gpu = tf.linalg.matmul(tf.constant(np0), tf.constant(np1)).numpy()

    tmp0 = [ret_np, ret_torch_cpu, ret_tf_cpu, ret_cp, ret_torch_gpu, ret_tf_gpu]
    str_name = ['numpy', 'torch-cpu', 'tf-cpu', 'cupy', 'torch-gpu', 'tf-gpu']
    data = np.array([[hfe(x,y) for y in tmp0] for x in tmp0])
    pd0 = pd.DataFrame(data, columns=str_name, index=str_name)
    print(pd0)


def demo_gpu_bandwidth():
    size = 2**28 #2**28 for 2GB
    num_gpu = 4
    device_list = [cp.cuda.Device(x) for x in range(num_gpu)]
    cp_list0 = []
    cp_list1 = []
    for device_i in device_list:
        with device_i:
            cp_list0.append(cp.random.uniform(size=size, dtype=cp.float64))
            cp_list1.append(cp.random.uniform(size=size, dtype=cp.float64))

    time_dict = dict()
    for ind0 in range(num_gpu):
        device_i = device_list[ind0]
        device_i.synchronize()
        time0 = time.time()
        cp.copyto(cp_list1[ind0], cp_list0[ind0])
        device_i.synchronize()
        time_dict[(ind0,ind0)] = time.time() - time0

    for ind0 in range(num_gpu-1):
        device_i = device_list[ind0]
        for ind1 in range(ind0+1, num_gpu):
            device_j = device_list[ind1]

            device_i.synchronize()
            device_j.synchronize()
            time0 = time.time()
            cp.copyto(cp_list0[ind1], cp_list0[ind0])
            device_i.synchronize()
            device_j.synchronize()
            time_dict[(ind0,ind1)] = time.time() - time0

            device_i.synchronize()
            device_j.synchronize()
            time0 = time.time()
            cp.copyto(cp_list0[ind0], cp_list0[ind1])
            device_i.synchronize()
            device_j.synchronize()
            time_dict[(ind1,ind0)] = time.time() - time0

    str_format = '{:10}' + '{:10}'*num_gpu
    print(str_format.format('GB/s', *['dst-{}'.format(x) for x in range(num_gpu)]))
    GB_size = size*8 / 2**30
    for ind0 in range(num_gpu):
        tmp0 = ['{:.5}'.format(GB_size / time_dict[(ind0,x)]) for x in range(num_gpu)]
        print(str_format.format(f'src-{ind0}', *tmp0))


if __name__=='__main__':
    demo_float32_diff_precision()

    # demo_gpu_bandwidth()
