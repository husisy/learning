import time
import queue
import numpy as np
import threading

class DummyThreadRunner00(threading.Thread):
    def __init__(self, q0, num_total):
        super().__init__()
        self.q0 = q0
        self.num_total = num_total
        self.data = []
    def run(self):
        for _ in range(self.num_total):
            x = self.q0.get()
            self.data.append(x)


def test_basic(N0=3):
    q0 = queue.Queue()
    worker = DummyThreadRunner00(q0, N0)
    worker.start()
    data = np.random.randint(0, 1000, size=N0).tolist()
    for x in data:
        q0.put(x)
    worker.join()
    assert all(x==y for x,y in zip(data,worker.data))


def test_numpy_no_copy(N0=3):
    q0 = queue.Queue()
    worker = DummyThreadRunner00(q0, N0)
    worker.start()
    data = [np.random.randint(0, 1000, size=3) for _ in range(N0)]
    for x in data:
        q0.put(x)
    worker.join()
    # no copy is made
    assert all(id(x)==id(y) for x,y in zip(data,worker.data))
    assert all(np.shares_memory(x,y) for x,y in zip(data,worker.data))


class DummyThreadRunner01(threading.Thread):
    def __init__(self, queue_m2w, queue_w2m, num_total):
        super().__init__()
        self.queue_m2w = queue_m2w #master to worker
        self.queue_w2m = queue_w2m #worker to master
        self.num_total = num_total
    def run(self):
        for _ in range(self.num_total):
            self.queue_w2m.put(self.queue_m2w.get())

def test_queue_in_queue_out(N0=3):
    queue_m2w = queue.Queue()
    queue_w2m = queue.Queue()
    worker = DummyThreadRunner01(queue_m2w, queue_w2m, N0)
    worker.start()
    data_send = [np.random.randint(0, 1000, size=3) for _ in range(N0)]
    for x in data_send:
        queue_m2w.put(x)
    data_receive = [queue_w2m.get() for _ in range(N0)]
    worker.join()
    # no copy is made
    assert all(id(x)==id(y) for x,y in zip(data_send,data_receive))
    assert all(np.shares_memory(x,y) for x,y in zip(data_send,data_receive))


class DummyThreadRunner02(threading.Thread):
    def __init__(self, list_, lock=None, num_repeat=100000):
        super().__init__()
        assert (len(list_)==1) and isinstance(list_[0], int)
        self.list_ = list_
        self.lock = lock
        self.num_repeat = num_repeat
    def run(self):
        def hf0():
            self.list_[0] = self.list_[0] + 1
            self.list_[0] = self.list_[0] - 1
        if self.lock is not None:
            for _ in range(self.num_repeat):
                self.lock.acquire()
                try:
                    hf0()
                finally:
                    self.lock.release()
        else:
            for _ in range(self.num_repeat):
                hf0()

def test_lock():
    value = 233
    num_worker = 4
    lock = threading.Lock()

    # with lock
    z0 = [value]
    worker_list = [DummyThreadRunner02(z0, lock) for _ in range(num_worker)]
    for x in worker_list:
        x.start()
    for x in worker_list:
        x.join()
    assert z0[0]==value

    # without lock
    z0 = [value]
    worker_list = [DummyThreadRunner02(z0) for _ in range(num_worker)]
    for x in worker_list:
        x.start()
    for x in worker_list:
        x.join()
    print('test_lock:', z0[0], value) #must time should be different
