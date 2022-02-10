import os
import time
import random
from filelock import FileLock
from concurrent.futures import ProcessPoolExecutor, as_completed


hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.mkdir(hf_file())


def func00(x):
    return x**2


def func01(ind1, x, lock_path):
    with FileLock(lock_path):
        tmp1 = random.random()
        print('pid={}, x={} start tp run'.format(os.getpid(), x))
        time.sleep(tmp1)
        print('pid={}, x={} has run {:.3} seconds'.format(os.getpid(), x, tmp1))
        return ind1, func00(x)

if __name__ == "__main__":
    arg_list = [2, 3, 5, 7, 11]
    ret_list = [None for _ in arg_list]
    lock_path = hf_file('test.lock')
    with ProcessPoolExecutor() as executor:
        future_job = [executor.submit(func01, ind0, x, lock_path) for ind0, x in enumerate(arg_list)]
        tmp0 = [x.result() for x in as_completed(future_job)]
        ret = [x[1] for x in sorted(tmp0, key=lambda x:x[0])]
        print('result: ', ret)
