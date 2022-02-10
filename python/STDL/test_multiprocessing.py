import os
import time
import numpy as np
os.environ['OMP_NUM_THREADS'] = '1'
import multiprocessing


hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())

def _pool_hf0(x):
    ret = x*x
    return ret

def _process_hf0(x, queue):
    queue.put((x, _pool_hf0(x)))

def test_process_basic():
    x0 = [1,2,3]
    ret_ = [_pool_hf0(x) for x in x0]
    queue = multiprocessing.Queue()
    process_list = [multiprocessing.Process(target=_process_hf0, args=(x,queue)) for x in x0]
    for x in process_list:
        x.start()
    ret0 = [queue.get() for _ in range(len(x0))]
    ret0 = [x[1] for x in sorted(ret0, key=lambda x:x[0])]
    assert all(x==y for x,y in zip(ret_,ret0))

    with multiprocessing.Pool(3) as pool:
        ret1 = pool.map(_pool_hf0, x0)
    assert all(x==y for x,y in zip(ret_,ret1))


class DummyProcessRunner00(multiprocessing.Process):
    def __init__(self, queue_m2w, queue_w2m, num_total):
        super().__init__()
        self.queue_m2w = queue_m2w
        self.queue_w2m = queue_w2m
        self.num_total = num_total
        self.x = None #x in main process will not change
    def run(self):
        for _ in range(self.num_total):
            self.x = self.queue_m2w.get()
            self.queue_w2m.put(self.x**2)

def test_process_queue00(N0=3):
    # data: master --> worker --> master
    queue_m2w = multiprocessing.Queue()
    queue_w2m = multiprocessing.Queue()
    worker = DummyProcessRunner00(queue_m2w, queue_w2m, N0)
    worker.start()
    data_send = [np.random.randint(0, 1000, size=3) for _ in range(N0)]
    ret_ = [x**2 for x in data_send]
    ret0 = []
    for x in data_send:
        queue_m2w.put(x)
        ret0.append(queue_w2m.get())
    worker.join()
    assert all(np.all(x==y) for x,y in zip(ret_,ret0))
    assert worker.x is None


class DummyProcessRunner01(multiprocessing.Process):
    def __init__(self, rank, barrier, filepath):
        assert not os.path.exists(filepath)
        super().__init__()
        self.barrier = barrier
        self.rank = rank
        self.filepath = filepath
    def run(self):
        if self.rank==0:
            time.sleep(1)
            with open(self.filepath, 'w') as fid:
                fid.write('hello world')
            self.barrier.wait()
        elif self.rank==1:
            self.barrier.wait()
            with open(self.filepath, 'r') as fid:
                assert fid.read()=='hello world'

def test_process_barrier():
    filepath = hf_file('_process_barrier.txt')
    if os.path.exists(filepath):
        os.remove(filepath)
    barrier = multiprocessing.Barrier(2)
    worker_list = [
        DummyProcessRunner01(0, barrier, filepath),
        DummyProcessRunner01(1, barrier, filepath)
    ]
    for x in worker_list:
        x.start()
    for x in worker_list:
        x.join()
    for x in worker_list:
        assert x.exitcode==0

# TODO test process lock
