import os
import time
import random
from filelock import FileLock
from concurrent.futures import ProcessPoolExecutor, as_completed


hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.mkdir(hf_file())


def demo_lock(func, _LOCK_FILE=hf_file('test.lock')):
    def wrapper(*args, **kw):
        with FileLock(_LOCK_FILE):
            tmp1 = random.random()
            print('pid={}, start tp run'.format(os.getpid()))
            ret = func(*args, **kw)
            time.sleep(tmp1)
            print('pid={} has run {:.3} seconds'.format(os.getpid(), tmp1))
        return ret
    return wrapper

@demo_lock
def func00(x):
    return x**2


# arg_list = [2, 3, 5, 7, 11]
# ret_list = [None for _ in arg_list]
# with ProcessPoolExecutor() as executor:
#     future_job = [executor.submit(func00, ind0, x) for ind0, x in enumerate(arg_list)]
#     tmp0 = [x.result() for x in as_completed(future_job)]
#     ret = [x[1] for x in sorted(tmp0, key=lambda x:x[0])]
#     print('result: ', ret)
