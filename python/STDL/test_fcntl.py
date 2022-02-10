import os
import fcntl
import random
import concurrent.futures

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())


class _DummyIOLockCounter:
    def __init__(self, filename, lock_complete=True, num_repeat=1000):
        lock_filename = filename + '.lock'
        assert os.path.exists(filename)
        self.filename = filename
        self.lock_fid = open(lock_filename, 'w')
        self.num_repeat = num_repeat
        self.lock_complete = lock_complete
    def increment(self, x):
        for _ in range(self.num_repeat):
            fcntl.lockf(self.lock_fid, fcntl.LOCK_EX)
            with open(self.filename, 'r') as fid:
                value = int(fid.read().strip()) + x
            if not self.lock_complete:
                #release then acquire
                fcntl.lockf(self.lock_fid, fcntl.LOCK_UN)
                fcntl.lockf(self.lock_fid, fcntl.LOCK_EX)
            with open(self.filename, 'w') as fid:
                fid.write(str(value))
            fcntl.lockf(self.lock_fid, fcntl.LOCK_UN)

def _process_i(filename, lock_complete, x, num_repeat):
    counter = _DummyIOLockCounter(filename, lock_complete, num_repeat)
    counter.increment(x)


def test_io_lock():
    filename = hf_file('io_lock.txt')
    with open(filename, 'w') as fid:
        fid.write('0')
    num_worker = 4
    num_repeat = 100
    x_list = [random.randint(-3,4) for _ in range(num_worker)]
    ret_ = sum(x_list) * num_repeat

    lock_complete = True
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker) as executor:
        job_list = [executor.submit(_process_i, filename, lock_complete, x, num_repeat) for x in x_list]
        for x in job_list:
            x.result()
    with open(filename) as fid:
        assert int(fid.read().strip())==ret_

    lock_complete = False
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker) as executor:
        job_list = [executor.submit(_process_i, filename, lock_complete, x, num_repeat) for x in x_list]
        for x in job_list:
            x.result()
    with open(filename) as fid:
        assert int(fid.read().strip())!=ret_
