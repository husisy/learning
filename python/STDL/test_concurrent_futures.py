import time
import random
import threading
import concurrent.futures


def hf_task_square(x):
    ret = x**2
    time.sleep(random.uniform(0.1,0.2))
    return ret


def test_concurrent_futures_ProcessPoolExecutor():
    arg_list = [random.randint(-100, 100) for _ in range(20)]
    ret_ = [x**2 for x in arg_list]

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        job_list = [executor.submit(hf_task_square, x) for x in arg_list]
        ret0 = [x.result() for x in job_list] #one-by-one
    assert all(x==y for x,y in zip(ret_,ret0))

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        job_list = [executor.submit(hf_task_square, x) for x in arg_list]
        id_list = [id(x) for x in job_list]
        id_to_result = {id(x):x.result() for x in concurrent.futures.as_completed(job_list)} #first in first out
        ret1 = [id_to_result[x] for x in id_list]
    assert all(x==y for x,y in zip(ret_,ret1))

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        ret2 = list(executor.map(hf_task_square, arg_list))
    assert all(x==y for x,y in zip(ret_,ret2))


class _DummyThreadingCounter:
    def __init__(self, lock=True, num_repeat=100000):
        self.count = 0
        self.num_repeat = num_repeat
        if lock:
            self.lock = threading.Lock()
        else:
            self.lock = None
    def increment(self, x):
        for _ in range(self.num_repeat):
            if self.lock is not None:
                self.lock.acquire()
                self.count += x
                self.lock.release()
            else:
                self.count += x

def _thread_safe_i(counter, x):
    counter.increment(x)

def test_thread_safe():
    num_thread = 4
    num_repeat = 100000
    x_list = [random.randint(-3, 4) for _ in range(num_thread)]
    ret_ = sum(x_list) * num_repeat

    counter = _DummyThreadingCounter(lock=True, num_repeat=num_repeat)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_thread) as executor:
        job_list = [executor.submit(_thread_safe_i, counter, x) for x in x_list]
        for x in job_list:
            x.result()
    assert counter.count==ret_

    counter = _DummyThreadingCounter(lock=False, num_repeat=num_repeat)
    x_list = [-2,-1,0,1]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_thread) as executor:
        job_list = [executor.submit(_thread_safe_i, counter, x) for x in x_list]
        for x in job_list:
            x.result()
    assert counter.count!=ret_ #when num_repeat is too small, may assert fail
