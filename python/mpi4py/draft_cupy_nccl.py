import numpy as np
from mpi4py import MPI
import cupy as cp
import cupy.cuda.nccl

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_world_size = mpi_comm.Get_size()

if __name__=='__main__':
    if mpi_world_size==2:
        pair_rank = {0:1, 1:0}[mpi_rank]
    elif mpi_world_size==4:
        pair_rank = {0:2, 1:3, 2:0, 3:1}[mpi_rank]
    else:
        assert False

    with cp.cuda.Device(mpi_rank):
        id_ = mpi_comm.bcast(cp.cuda.nccl.get_unique_id() if mpi_rank==0 else None, root=0)
        comm = cp.cuda.nccl.NcclCommunicator(mpi_world_size, id_, mpi_rank)
        cp0 = cp.array([mpi_rank]*3, dtype=cp.float64)
        cp1 = cp.empty(3, dtype=cp.float64)

        print(f'[rank={mpi_rank}] pair_rank:', pair_rank)
        if mpi_rank < pair_rank:
            comm.send(cp0.data.ptr, 3, cp.cuda.nccl.NCCL_FLOAT64, pair_rank, cp.cuda.Stream.null.ptr)
            comm.recv(cp1.data.ptr, 3, cp.cuda.nccl.NCCL_FLOAT64, pair_rank, cp.cuda.Stream.null.ptr)
        else:
            comm.recv(cp1.data.ptr, 3, cp.cuda.nccl.NCCL_FLOAT64, pair_rank, cp.cuda.Stream.null.ptr)
            comm.send(cp0.data.ptr, 3, cp.cuda.nccl.NCCL_FLOAT64, pair_rank, cp.cuda.Stream.null.ptr)
        print(f'[rank={mpi_rank}] cp0:', cp0.get())
        print(f'[rank={mpi_rank}] cp1:', cp1.get())
