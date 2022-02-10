# TODO test concurrents.future with different environment

# TODO how to compare multiprocessing and openmp: mangy repeats work multiprocessing.Barrier
import os
import time
import pickle
import numpy as np
import contextlib
import multiprocessing

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())

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


class DummyProcessRunner00(multiprocessing.Process):
    def __init__(self, N0, num_repeat, rank, queue_w2m, barrier=None):
        super().__init__()
        self.N0 = N0
        self.num_repeat = num_repeat
        self.rank = rank
        self.barrier = barrier
        self.queue_w2m = queue_w2m
    def run(self):
        np0 = np.random.randn(self.N0, self.N0)
        np1 = np.random.randn(self.N0, self.N0)
        np2 = np.zeros_like(np0)
        if self.barrier is not None:
            self.barrier.wait()
        t0 = time.time()
        for _ in range(self.num_repeat):
            # np.sin(np0, out=np2)
            np.matmul(np0, np1, out=np2)
        t1 = time.time() - t0
        if self.barrier is not None:
            self.barrier.wait()
        t2 = time.time() - t0
        self.queue_w2m.put((self.rank,t1,t2))


if __name__=='__main__':
    # total_thread = 24
    # N0 = 2048
    # total_repeat = total_thread*8
    # num_process_list = [1,2,4,8,12,24]
    # assert all(total_thread%x==0 for x in num_process_list)
    # num_thread_list = [total_thread//x for x in num_process_list]
    # num_repeat_list = [total_repeat//x for x in num_process_list]

    N0 = 2048
    num_process_list = [1,12,24]
    num_thread_list = [1,1,1]
    num_repeat_list = [16,16,16]

    time_info = []
    for num_thread,num_process,num_repeat in zip(num_thread_list,num_process_list,num_repeat_list):
        print(f'num_thread={num_thread},num_process={num_process},num_repeat={num_repeat}')
        with subprocess_env(OMP_NUM_THREADS=str(num_thread)):
            queue_w2m = multiprocessing.Queue()
            barrier = multiprocessing.Barrier(num_process) if (num_process>1) else None
            worker_list = [DummyProcessRunner00(N0, num_repeat, x, queue_w2m, barrier) for x in range(num_process)]
            for x in worker_list:
                x.start()
            for x in worker_list:
                x.join()
            z0 = [queue_w2m.get() for _ in range(num_process)]
            assert {x[0] for x in z0}==set(range(num_process))
            time_info.append(sorted(z0, key=lambda x:x[0]))

    with open(hf_file('scipy_openmp_performance.pkl'), 'wb') as fid:
        tmp0 = {
            'N0': N0,
            'num_process_list': num_process_list,
            'num_thread_list': num_thread_list,
            'num_repeat_list': num_repeat_list,
            'time_info': time_info,
        }
        pickle.dump(tmp0, fid)

    print('matrix-size:', N0)
    for x in time_info:
        tmp0 = max(y[2] for y in x) - min(y[2] for y in x)
        if tmp0 > 0.05:
            print('WARNING, large statistics error', tmp0)

    z0 = [np.mean(np.array(x)[:,1:], axis=0) for x in time_info]
    print('| `NumProcess x NumThread` | average-time (s) | sync-time (s) |')
    print('| :-: | :-: | :-: |')
    for num_process,num_thread,(time0,time1) in zip(num_process_list,num_thread_list,z0):
        print(f'| `{num_process}x{num_thread}` | `{time0:.3}` | `{time1:.3}` |')

