import os
import time
import numpy as np
import concurrent.futures
import contextlib

@contextlib.contextmanager
def subprocess_env(**kwargs):
    old_environment = {k:os.environ.get(k) for k in kwargs.keys()}
    for k,v in kwargs.items():
        os.environ[k] = v
    yield None
    for k,v in old_environment.items():
        if v is None:
            del os.environ[k]
        else:
            os.environ[k] = v


def subprocess_work(hf0, args=(), envs=None):
    if envs is None:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            ret = executor.submit(hf0, *args).result()
    else:
        assert isinstance(envs, dict)
        with subprocess_env(**envs):
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                ret = executor.submit(hf0, *args).result()
    return ret


def _env_OMP_NUM_THREADS_matmul(N0=4096, num_repeat=5):
    print('OMP_NUM_THREADS: ', os.environ.get('OMP_NUM_THREADS'))
    np0 = np.random.randn(N0,N0)
    np1 = np.random.randn(N0,N0)
    np2 = np.zeros_like(np0)
    time_info = []
    for _ in range(num_repeat):
        t0 = time.time()
        np.matmul(np0, np1, out=np2)
        time_info.append(time.time() - t0)
    time_info = np.array(time_info)
    return time_info

def demo_env_OMP_NUM_THREADS():
    print('# demo_env_OMP_NUM_THREADS')
    print('main-process call matmul')
    print(_env_OMP_NUM_THREADS_matmul())

    print('subprocess call matmul')
    envs = {'OMP_NUM_THREADS':'1'}
    print(subprocess_work(_env_OMP_NUM_THREADS_matmul, envs=envs))



def _env_OMP_NUM_THREADS_1248_matmul(N_list, num_repeat=5):
    time_info = []
    for N in N_list:
        time_i = []
        for _ in range(num_repeat):
            np0 = np.random.randn(N,N)
            np1 = np.random.randn(N,N)
            np2 = np.zeros_like(np0)
            t0 = time.time()
            np.matmul(np0, np1, out=np2)
            time_i.append(time.time() - t0)
        time_info.append(time_i)
    time_info = np.array([np.array(x).mean() for x in time_info])
    return time_info


def demo_env_OMP_NUM_THREADS_1248():
    num_threads_list = [1,2,4,8,16,24,32]
    N_list = [64, 256, 1024, 4096]

    z0 = []
    for num_thread in num_threads_list:
        envs = {'OMP_NUM_THREADS':str(num_thread)}
        z0.append(subprocess_work(_env_OMP_NUM_THREADS_1248_matmul, args=(N_list,), envs=envs))
    z0 = np.stack(z0)
    speed_up_ratio = z0[0]/z0

    tmp0 = ' | '.join(str(x) for x in N_list)
    print(f'| matrix-size | {tmp0} |')
    print('| ' + ' | '.join([':-:']*(len(N_list)+1)) + ' |')
    for x,speed_i in zip(num_threads_list,speed_up_ratio):
        tmp0 = ' | '.join(f'`{x:.3}`' for x in speed_i)
        print(f'| thread={x} | {tmp0} |')

if __name__=='__main__':
    # demo_env_OMP_NUM_THREADS()

    demo_env_OMP_NUM_THREADS_1248()
