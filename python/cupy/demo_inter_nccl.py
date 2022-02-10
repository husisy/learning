# proc0 dgx162 export NCCL_SOCKET_IFNAME=enp2s0f1
import cupy as cp
import cupy.cuda.nccl
import pickle
rank = 0
cp.cuda.Device(rank).use()
id_ = cp.cuda.nccl.get_unique_id() #(tuple,int)
with open('id_.pkl', 'wb') as fid:
    pickle.dump(id_,fid)
comm = cp.cuda.nccl.NcclCommunicator(2, id_, rank)
cp0 = cp.array([1,2,3], dtype=cp.float32)
comm.send(cp0.data.ptr, cp0.shape[0], cp.cuda.nccl.NCCL_FLOAT32, 1, cp.cuda.Stream.null.ptr)


# proc1 dgx163 export NCCL_SOCKET_IFNAME=enp2s0f1
import cupy as cp
import cupy.cuda.nccl
import pickle
rank = 1
cp.cuda.Device(rank).use()
with open('id_.pkl','rb') as fid: #copy from proc0
    id_ = pickle.load(fid)
comm = cp.cuda.nccl.NcclCommunicator(2, id_, rank)
cp0 = cp.array([0,0,0], dtype=cp.float32)
comm.recv(cp0.data.ptr, cp0.shape[0], cp.cuda.nccl.NCCL_FLOAT32, 0, cp.cuda.Stream.null.ptr)
