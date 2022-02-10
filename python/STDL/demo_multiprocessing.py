import os
import time
import socket
import random
import threading
import multiprocessing
import multiprocessing.managers
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


def processing_kernel_time(hf0, num_worker, args=()):
    worker_list = [multiprocessing.Process(target=hf0, args=args) for _ in range(num_worker)]
    t0 = time.time()
    for x in worker_list:
        x.start()
    for x in worker_list:
        x.join()
    ret = time.time() - t0
    return ret


def _pool_advanced_hf0(x):
    ret = x*x
    return ret

# TODO
def demo_pool_advanced():
    with multiprocessing.Pool(processes=4) as pool:
        print('map():', pool.map(_pool_advanced_hf0, range(10)))
        print('imap_unordered():', list(pool.imap_unordered(_pool_advanced_hf0, range(10)))) #in arbitrary order

        tmp0 = pool.apply_async(_pool_advanced_hf0, (20,))
        print('apply_async():', tmp0.get(timeout=1))

        tmp0 = [pool.apply_async(os.getpid, ()) for _ in range(4)]
        print('list-apply_async():', [x.get(timeout=1) for x in tmp0])

        tmp0 = pool.apply_async(time.sleep, (10,))
        try:
            print(tmp0.get(timeout=1))
        except multiprocessing.context.TimeoutError:
            print('multiprocessing.context.TimeoutError')


def _no_GIL_issue_worker(N0=10000000):
    ret = 0.233
    t0 = time.time()
    for x in range(N0):
        ret = ret + N0
    t1 = time.time() - t0
    print(f'[worker] time={t1:.3} seconds')

def demo_no_GIL_issue():
    num_worker_list = [1,2,4]
    for num_worker in num_worker_list:
        t0 = processing_kernel_time(_no_GIL_issue_worker, num_worker)
        print(f'[num_worker={num_worker}] time={t0:.3} seconds')
    # [worker] time=0.52 seconds
    # [num_worker=1] time=0.526 seconds
    # [worker] time=0.497 seconds
    # [worker] time=0.526 seconds
    # [num_worker=2] time=0.531 seconds
    # [worker] time=0.532 seconds
    # [worker] time=0.537 seconds
    # [worker] time=0.541 seconds
    # [worker] time=0.542 seconds
    # [num_worker=4] time=0.552 seconds


def _numpy_processing_performance_i(N0=1024, num_repeat=100):
    # num_repeat=100 takes up almost 5 seconds
    np0 = np.random.randn(N0, N0)
    np1 = np.random.randn(N0, N0)
    np2 = np.zeros_like(np0)
    t0 = time.time()
    for _ in range(num_repeat):
        _ = np.matmul(np0, np1, out=np2)
    t1 = time.time() - t0
    print(f'[worker] time={t1:.3} seconds')

def demo_numpy_processing_performance():
    num_worker_list = [1,2,4,8]

    for num_worker in num_worker_list:
        t0 = processing_kernel_time(_numpy_processing_performance_i, num_worker)
        print(f'[num_worker={num_worker}] time={t0:.3} seconds')
    # [worker] time=5.15 seconds
    # [num_worker=1] time=5.23 seconds
    # [worker] time=5.16 seconds
    # [worker] time=5.22 seconds
    # [num_worker=2] time=5.31 seconds
    # [worker] time=5.33 seconds
    # [worker] time=5.34 seconds
    # [worker] time=5.74 seconds
    # [worker] time=5.77 seconds
    # [num_worker=4] time=5.85 seconds
    # [worker] time=6.56 seconds
    # [worker] time=6.57 seconds
    # [worker] time=6.59 seconds
    # [worker] time=6.76 seconds
    # [worker] time=6.82 seconds
    # [worker] time=6.92 seconds
    # [worker] time=6.94 seconds
    # [worker] time=6.99 seconds
    # [num_worker=8] time=7.08 seconds


def _process_io_read_speed_i(logdir):
    t0 = time.time()
    for x in os.listdir(logdir):
        with open(os.path.join(logdir,x), 'rb') as fid:
            _ = len(fid.read())
    t1=  time.time() - t0
    print(f'[worker] time={t1:.3} seconds')

def _process_np_fromfile_speed_i(logdir):
    t0 = time.time()
    for x in os.listdir(logdir):
        _ = np.fromfile(os.path.join(logdir,x), dtype=np.uint8).size
    t1=  time.time() - t0
    print(f'[worker] time={t1:.3} seconds')

# maybe ssd + raid?
def demo_process_io_read_speed():
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
        t0 = processing_kernel_time(_process_io_read_speed_i, num_worker, args=(logdir,))
        print(f'[num_worker={num_worker}] time={t0:.3} seconds')
    # [worker] time=0.381 seconds
    # [num_worker=1] time=0.402 seconds
    # [worker] time=0.447 seconds
    # [worker] time=0.449 seconds
    # [num_worker=2] time=0.471 seconds
    # [worker] time=0.416 seconds
    # [worker] time=0.433 seconds
    # [worker] time=0.432 seconds
    # [worker] time=0.451 seconds
    # [num_worker=4] time=0.478 seconds
    # [worker] time=0.522 seconds
    # [worker] time=0.539 seconds
    # [worker] time=0.543 seconds
    # [worker] time=0.568 seconds
    # [worker] time=0.569 seconds
    # [worker] time=0.574 seconds
    # [worker] time=0.577 seconds
    # [worker] time=0.572 seconds
    # [num_worker=8] time=0.601 seconds

    for num_worker in num_worker_list:
        t0 = processing_kernel_time(_process_np_fromfile_speed_i, num_worker, args=(logdir,))
        print(f'[num_worker={num_worker}] time={t0:.3} seconds')
    # [worker] time=0.437 seconds
    # [num_worker=1] time=0.441 seconds
    # [worker] time=0.46 seconds
    # [worker] time=0.537 seconds
    # [num_worker=2] time=0.543 seconds
    # [worker] time=0.564 seconds
    # [worker] time=0.562 seconds
    # [worker] time=0.568 seconds
    # [worker] time=0.566 seconds
    # [num_worker=4] time=0.578 seconds
    # [worker] time=0.548 seconds
    # [worker] time=0.546 seconds
    # [worker] time=0.551 seconds
    # [worker] time=0.555 seconds
    # [worker] time=0.544 seconds
    # [worker] time=0.553 seconds
    # [worker] time=0.576 seconds
    # [worker] time=0.576 seconds
    # [num_worker=8] time=0.594 seconds


def _process_terminate_i(queue):
    x = queue.get()
    ret = x**2
    return ret

class MyWorkerList:
    def __init__(self, worker_list):
        self.worker_list = worker_list
    def __getitem(self, index):
        return self.worker_list[index]
    def __iter__(self):
        yield from self.worker_list
    def __del__(self):
        for x in self.worker_list:
            if not x.is_alive():
                x.terminate()

def demo_process_terminate():
    num_worker = 4
    queue_list = [multiprocessing.Queue() for _ in range(num_worker)]
    tmp0 = [multiprocessing.Process(target=_process_terminate_i, args=(x,)) for x in queue_list]
    worker_list = MyWorkerList(tmp0)
    for x in worker_list:
        x.start()
    # for x in queue_list:
    #     x.put(233)
    # use MyWokerList, so x.terminate will be automatically called even the worker is not finished


def get_free_port():
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

class QueueServerManager(multiprocessing.managers.BaseManager):
    pass

class QueueClientManager(multiprocessing.managers.BaseManager):
    pass

class QueueServerWorker(threading.Thread):
    def __init__(self, queue_c2s_list, queue_s2c_list):
        super().__init__()
        self.queue_c2s_list = queue_c2s_list
        self.queue_s2c_list = queue_s2c_list
        self.world_size = len(self.queue_c2s_list)
    def run(self):
        tag_quit = [False for _ in range(self.world_size)]
        while True:
            for ind0 in range(self.world_size):
                if not tag_quit[ind0]:
                    data = self.queue_c2s_list[ind0].get()
                    if data=='quit':
                        tag_quit[ind0] = True
                        self.queue_c2s_list[ind0].close()
                        self.queue_s2c_list[ind0].close()
                    else:
                        self.queue_s2c_list[ind0].put(data[::-1])
            if all(tag_quit):
                break

class QueueClientWorker(multiprocessing.Process):
    def __init__(self, ip, port, authkey, rank, num_data):
        super().__init__()
        QueueClientManager.register('get_queue_c2s')
        QueueClientManager.register('get_queue_s2c')
        self.manager = QueueClientManager(address=(ip, port), authkey=authkey)
        self.manager.connect()
        self.rank = rank
        self.queue_c2s = self.manager.get_queue_c2s(rank)
        self.queue_s2c = self.manager.get_queue_s2c(rank)
        self.data = [f'{rank}-{100+x}' for x in range(num_data)]
    def run(self):
        for x in self.data:
            print(f'[rank={self.rank}][put] {x}')
            self.queue_c2s.put(x)
            tmp0 = self.queue_s2c.get()
            print(f'[rank={self.rank}][get] {tmp0}')
        self.queue_c2s.put('quit')

def demo_remote_connect_queue():
    world_size = 4
    ip = '127.0.0.1'
    authkey = b'233333'
    num_data = 3
    port = get_free_port()

    queue_c2s_list = [multiprocessing.Queue() for x in range(world_size)]
    queue_s2c_list = [multiprocessing.Queue() for x in range(world_size)]
    QueueServerManager.register('get_queue_c2s', callable=lambda x: queue_c2s_list[x])
    QueueServerManager.register('get_queue_s2c', callable=lambda x: queue_s2c_list[x])
    server_manager = QueueServerManager(address=(ip, port), authkey=authkey)
    server_manager.start()
    server_worker = QueueServerWorker(queue_c2s_list, queue_s2c_list)
    server_worker.start()
    client_worker_list = [QueueClientWorker(ip, port, authkey, x, num_data) for x in range(world_size)]
    for x in client_worker_list:
        x.start()
    for x in client_worker_list:
        x.join()


class PipeServerManager(multiprocessing.managers.BaseManager):
    pass

class PipeClientManager(multiprocessing.managers.BaseManager):
    pass

class PipeServerWorker(threading.Thread):
    def __init__(self, pipe_server_list):
        super().__init__()
        self.pipe_server_list = pipe_server_list
        self.world_size = len(self.pipe_server_list)
    def run(self):
        tag_quit = [False for _ in range(self.world_size)]
        while True:
            for ind0 in range(self.world_size):
                if not tag_quit[ind0]:
                    data = self.pipe_server_list[ind0].recv()
                    if data=='quit':
                        tag_quit[ind0] = True
                        self.pipe_server_list[ind0].close()
                    else:
                        self.pipe_server_list[ind0].send(data[::-1])
            if all(tag_quit):
                break

class PipeClientWorker(multiprocessing.Process):
    def __init__(self, ip, port, authkey, rank, num_data):
        super().__init__()
        PipeClientManager.register('get_pipe')
        self.manager = PipeClientManager(address=(ip, port), authkey=authkey)
        self.manager.connect()
        self.rank = rank
        self.pipe_client = self.manager.get_pipe(rank)
        self.data = [f'{rank}-{100+x}' for x in range(num_data)]
    def run(self):
        for x in self.data:
            print(f'[rank={self.rank}][send] {x}')
            self.pipe_client.send(x)
            tmp0 = self.pipe_client.recv()
            print(f'[rank={self.rank}][recv] {tmp0}')
        self.pipe_client.send('quit')
        self.pipe_client.close()

def demo_remote_connect_pipe():
    world_size = 4
    ip = '127.0.0.1'
    authkey = b'233333'
    num_data = 3
    port = get_free_port()

    tmp0 = [multiprocessing.Pipe() for _ in range(world_size)]
    pipe_server_list = [x[0] for x in tmp0]
    pipe_client_list = [x[1] for x in tmp0]
    PipeServerManager.register('get_pipe', callable=lambda x: pipe_client_list[x])
    server_manager = PipeServerManager(address=(ip, port), authkey=authkey)
    server_manager.start()
    server_worker = PipeServerWorker(pipe_server_list)
    server_worker.start()
    client_worker_list = [PipeClientWorker(ip, port, authkey, x, num_data) for x in range(world_size)]
    for x in client_worker_list:
        x.start()
    for x in client_worker_list:
        x.join()



if __name__=='__main__':
    # demo_pool_advanced()

    # demo_no_GIL_issue()

    # demo_numpy_processing_performance()

    # demo_process_io_read_speed()

    # demo_process_terminate()

    # demo_remote_connect_queue()

    demo_remote_connect_pipe()
