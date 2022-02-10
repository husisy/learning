import time
import numpy as np
from mpi4py import MPI
from datetime import datetime

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

# TODO sync
def one_print(*args, rank_id=0):
    if mpi_rank==rank_id:
        print('[rank={}]'.format(rank_id), *args)

def demo00():
    if mpi_size < 2:
        return
    one_print('# demo00')

    if mpi_rank==0:
        data = {'a':7, 'b':3.14} #using pickle
        mpi_comm.send(data, dest=1, tag=233) #not-block
        print('[rank=0]: send data:', data)
    elif mpi_rank==1:
        data = mpi_comm.recv(source=0, tag=233) #block
        print('[rank=1]: receive data:', data)
    else:
        print('[rank={}] do nothing'.format(mpi_rank))


def demo_blocking_p2p():
    if mpi_size < 2:
        return
    one_print('# demo_blocking_p2p')

    if mpi_rank==0:
        data = 233
        time.sleep(1)
        print('[rank=0] start at', datetime.now())
        mpi_comm.send(data, dest=1, tag=234) #not block
        print('[rank=0] finish at', datetime.now())
    elif mpi_rank==1:
        print('[rank=1] start at', datetime.now())
        data = mpi_comm.recv(source=0, tag=234) #block
        print('[rank=1] finish at:', datetime.now())


# TODO non-blocking communication, https://mpi4py.readthedocs.io/en/stable/tutorial.html#point-to-point-communication

# TODO data_s, data_r

def demo_blocking_p2p_numpy():
    if mpi_size < 2:
        return
    one_print('# demo_blocking_p2p_numpy')
    if mpi_rank == 0:
        data = np.array([2,23,233], dtype=np.int64)
        print('[rank=0] send data:', data)
        mpi_comm.Send(data, dest=1, tag=235)
        # mpi_comm.Send([data, MPI.INT], dest=1, tag=235)
    elif mpi_rank == 1:
        data = np.empty(3, dtype=np.int64)
        mpi_comm.Recv(data, source=0, tag=235)
        # mpi_comm.Recv([data, MPI.INT], source=0, tag=235)
        print('[rank=1] receive data:', data)


def demo_broadcast():
    if mpi_size < 2:
        return
    one_print('# demo_broadcast')
    data = None
    if mpi_rank == 0:
        data = [2,23,233]
    data = mpi_comm.bcast(data, root=0)
    print('[rank={}] data: {}'.format(mpi_rank, data))


def demo_broadcast_numpy():
    if mpi_size < 2:
        return
    one_print('# demo_broadcast_numpy')
    if mpi_rank==0:
        data = np.array([2,23,233], dtype=np.int64)
    else:
        data = np.empty(3, dtype=np.int64)
    mpi_comm.Bcast(data, root=0)
    print('[rank={}] type={}, data={}'.format(mpi_rank, type(data), data))


def demo_scatter_list():
    if mpi_size < 2:
        return
    one_print('# demo_scatter_list')
    data = None
    if mpi_rank == 0:
        data = [x**2 for x in range(mpi_size)]
    data = mpi_comm.scatter(data, root=0)
    print('[rank={}] data: {}'.format(mpi_rank, data))


def demo_scatter_numpy():
    if mpi_size < 2:
        return
    one_print('# demo_scatter_numpy')
    data_s = 0
    if mpi_rank==0:
        data_s = (np.arange(mpi_size, dtype=np.int64)[:,np.newaxis]*np.ones(6, dtype=np.int64)).reshape(-1,3,2)
    data_r = np.empty((3,2), dtype=np.int64)
    mpi_comm.Scatter(data_s, data_r, root=0)
    print('[rank={}] data: {}'.format(mpi_rank, data_r.reshape(-1)))


def demo_gather_list():
    if mpi_size < 2:
        return
    one_print('# demo_gather_list')
    data_s = mpi_rank**2
    data_r = mpi_comm.gather(data_s, root=0)
    if mpi_rank==0:
        print('[rank=0] type={}, data={}'.format(type(data_r), data_r)) #list of int
    else:
        assert data_r is None


def demo_gather_numpy():
    if mpi_size < 2:
        return
    one_print('# demo_gather_numpy')
    data_s = np.ones(3, dtype=np.int64)*mpi_rank
    data_r = None
    if mpi_rank==0:
        data_r = np.zeros((mpi_size,3), dtype=np.int64)
    mpi_comm.Gather(data_s, data_r, root=0)
    if mpi_rank==0:
        print('[rank=0] type={}, data={}'.format(type(data_r), data_r.reshape(-1)))
    else:
        assert data_r is None


def demo_allgather_numpy():
    if mpi_size < 2:
        return
    one_print('# demo_allgather_numpy')
    data_s = np.ones(3, dtype=np.int64)*mpi_rank
    data_r = np.zeros((mpi_size,3), dtype=np.int64)
    mpi_comm.Allgather(data_s, data_r)
    print('[rank={}] data: {}'.format(mpi_rank, data_r.reshape(-1)))


# mpiexec -n 4 python draft00.py
if __name__=='__main__':
    demo00()
    # demo_blocking_p2p()
    # demo_blocking_p2p_numpy()
    # demo_broadcast()
    # demo_broadcast_numpy()
    # demo_scatter_list()
    # demo_scatter_numpy()
    # demo_gather_list()
    # demo_gather_numpy()
    # demo_allgather_numpy()
