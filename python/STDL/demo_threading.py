import os
import time
import random
import threading
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np


def next_tbd_dir(dir0='tbd00', maximum_int=100000):
    if not os.path.exists(dir0):
        os.makedirs(dir0)
    tmp1 = [x for x in os.listdir(dir0) if x[:3]=='tbd']
    exist_set = {x[3:] for x in tmp1}
    while True:
        tmp1 = str(random.randint(1,maximum_int))
        if tmp1 not in exist_set:
            break
    tbd_dir = os.path.join(dir0, 'tbd'+tmp1)
    os.mkdir(tbd_dir)
    return tbd_dir


def thread_kernel_time(hf0, num_worker, args=()):
    worker_list = [threading.Thread(target=hf0, args=args) for _ in range(num_worker)]
    t0 = time.time()
    for x in worker_list:
        x.start()
    for x in worker_list:
        x.join()
    ret = time.time() - t0
    return ret


def hf_task00():
    print('[hf_task00] current-thread-name:', threading.current_thread().name)


def demo_thread_name():
    print('# demo_thread_name')
    print('[startup] current-thread-name:', threading.current_thread().name)
    t = threading.Thread(target=hf_task00, name='Call(hf_task00)')
    t.start()
    t.join()
    print('[after join] current-thread-name:', threading.current_thread().name)


def _GIL_issue_worker(N0=10000000):
    ret = 0.233
    t0 = time.time()
    for x in range(N0):
        ret = ret + N0
    t1 = time.time() - t0
    print(f'[worker] time={t1:.3} seconds')

def demo_GIL_issue():
    num_worker_list = [1,2,4]
    for num_worker in num_worker_list:
        t0 = thread_kernel_time(_GIL_issue_worker, num_worker)
        print(f'[num_worker={num_worker}] time={t0:.3} seconds')
    # [worker] time=0.52 seconds
    # [num_worker=1] time=0.521 seconds
    # [worker] time=3.07 seconds
    # [worker] time=3.08 seconds
    # [num_worker=2] time=3.1 seconds
    # [worker] time=5.9 seconds
    # [worker] time=6.11 seconds
    # [worker] time=6.15 seconds
    # [worker] time=6.15 seconds
    # [num_worker=4] time=6.19 seconds

def _numpy_thread_performance_i(N0=1024, num_repeat=100):
    # num_repeat=100 takes up almost 5 seconds
    np0 = np.random.randn(N0, N0)
    np1 = np.random.randn(N0, N0)
    np2 = np.zeros_like(np0)
    t0 = time.time()
    for _ in range(num_repeat):
        _ = np.matmul(np0, np1, out=np2)
    t1 = time.time() - t0
    print(f'[worker] time={t1:.3} seconds')

def demo_numpy_thread_performance():
    num_worker_list = [1,2,4,8]

    for num_worker in num_worker_list:
        t0 = thread_kernel_time(_numpy_thread_performance_i, num_worker)
        print(f'[num_worker={num_worker}] time={t0:.3} seconds')
    # [worker] time=5.16 seconds
    # [num_worker=1] time=5.24 seconds
    # [worker] time=5.13 seconds
    # [worker] time=5.18 seconds
    # [num_worker=2] time=5.33 seconds
    # [worker] time=5.35 seconds
    # [worker] time=5.28 seconds
    # [worker] time=5.64 seconds
    # [worker] time=5.67 seconds
    # [num_worker=4] time=5.95 seconds
    # [worker] time=6.28 seconds
    # [worker] time=6.43 seconds
    # [worker] time=6.75 seconds
    # [worker] time=6.39 seconds
    # [worker] time=6.72 seconds
    # [worker] time=6.76 seconds
    # [worker] time=7.02 seconds
    # [worker] time=6.96 seconds
    # [num_worker=8] time=7.59 seconds

def _thread_io_read_speed_i(logdir):
    t0 = time.time()
    for x in os.listdir(logdir):
        with open(os.path.join(logdir,x), 'rb') as fid:
            _ = len(fid.read())
    t1=  time.time() - t0
    print(f'[worker] time={t1:.3} seconds')

def _thread_np_fromfile_speed_i(logdir):
    t0 = time.time()
    for x in os.listdir(logdir):
        _ = np.fromfile(os.path.join(logdir,x), dtype=np.uint8).size
    t1=  time.time() - t0
    print(f'[worker] time={t1:.3} seconds')

# maybe ssd + raid?
def demo_thread_io_read_speed():
    logdir = next_tbd_dir() #/zcdata/tbd00
    num_file = 4000
    num_byte = 2**19 #512KB
    rng = np.random.default_rng()
    file_list = [os.path.join(logdir, f'{x}.byte') for x in range(num_file)]
    for file_i in file_list:
        with open(file_i, 'wb') as fid:
            fid.write(rng.integers(0, 256, size=num_byte, dtype=np.uint8).tobytes())
    num_worker_list = [1,2,4,8]

    for num_worker in num_worker_list:
        t0 = thread_kernel_time(_thread_io_read_speed_i, num_worker, args=(logdir,))
        print(f'[num_worker={num_worker}] time={t0:.3} seconds')
    # [worker] time=0.388 seconds
    # [num_worker=1] time=0.404 seconds
    # [worker] time=0.625 seconds
    # [worker] time=0.626 seconds
    # [num_worker=2] time=0.657 seconds
    # [worker] time=0.799 seconds
    # [worker] time=0.776 seconds
    # [worker] time=0.756 seconds
    # [worker] time=0.782 seconds
    # [num_worker=4] time=0.893 seconds
    # [worker] time=3.69 seconds
    # [worker] time=3.79 seconds
    # [worker] time=3.8 seconds
    # [worker] time=3.89 seconds
    # [worker] time=3.85 seconds
    # [worker] time=3.87 seconds
    # [worker] time=3.92 seconds
    # [worker] time=3.84 seconds
    # [num_worker=8] time=4.3 seconds

    for num_worker in num_worker_list:
        t0 = thread_kernel_time(_thread_np_fromfile_speed_i, num_worker, args=(logdir,))
        print(f'[num_worker={num_worker}] time={t0:.3} seconds')
    # [worker] time=0.41 seconds
    # [num_worker=1] time=0.411 seconds
    # [worker] time=0.467 seconds
    # [worker] time=0.473 seconds
    # [num_worker=2] time=0.474 seconds
    # [worker] time=0.982 seconds
    # [worker] time=0.982 seconds
    # [worker] time=0.982 seconds
    # [worker] time=1.03 seconds
    # [num_worker=4] time=1.03 seconds
    # [worker] time=4.68 seconds
    # [worker] time=4.91 seconds
    # [worker] time=4.94 seconds
    # [worker] time=4.95 seconds
    # [worker] time=4.96 seconds
    # [worker] time=4.96 seconds
    # [worker] time=5.0 seconds
    # [worker] time=5.01 seconds
    # [num_worker=8] time=5.01 seconds

if __name__=='__main__':
    # demo_thread_name()

    # demo_GIL_issue()

    # demo_numpy_thread_performance()

    demo_thread_io_read_speed()
