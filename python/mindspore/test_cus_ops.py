import random
import numpy as np
import mindspore as ms
import pytest

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

from utils import detect_target_device

ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target=detect_target_device())


def np_unfold(npx, kernel_size_list, stride_list=None, padding_list=None):
    N0 = npx.ndim
    assert (N0>2) and (len(kernel_size_list)==(N0-2))
    if stride_list is None:
        stride_list = (1,)*(N0-2)
    else:
        assert len(stride_list) == N0-2
    if padding_list is not None:
        assert (len(padding_list)==N0-2) and all(len(x)==2 for x in padding_list)
        npx = np.pad(npx, [(0,0),(0,0)] + padding_list)
    old_shape = npx.shape
    old_strides = npx.strides
    hf_out_size = lambda in_size,kernel_size,stride: 1 + (in_size-(kernel_size-1)-1) // stride
    out_size = [hf_out_size(x,y,z) for x,y,z in zip(old_shape[2:],kernel_size_list,stride_list)]
    new_shape = old_shape[:2] + tuple(y for x in zip(out_size,kernel_size_list) for y in x)
    tmp0 = [(x*y,x) for x,y in zip(old_strides[2:],stride_list)]
    new_strides = old_strides[:2] + tuple(y for x in tmp0 for y in x)
    ret = np.lib.stride_tricks.as_strided(npx, shape=new_shape, strides=new_strides).copy()
    assert not np.any(np.isnan(ret).reshape(-1))
    tmp0 = [0] + list(range(1,2*N0-2,2)) + list(range(2,2*N0-2,2))
    tmp1 = [old_shape[0], old_shape[1]*np.prod(kernel_size_list), np.prod(out_size)]
    ret233 = ret.transpose(*tmp0).reshape(tmp1)
    return ret233


@pytest.mark.skipif(ms.context.get_context('device_target')!='GPU', reason='GPU only')
def test_Im2Col(batch_size=2, in_channel=3, in_height=57, in_width=59, kernel_size=(4,5), stride=(2,3)):
    npx = np.random.randn(batch_size, in_channel, in_height, in_width).astype(np.float32)

    hf_out_size = lambda in_size,kernel_size,stride: 1 + (in_size-(kernel_size-1)-1) // stride
    tmp0 = hf_out_size(in_height,kernel_size[0],stride[0]), hf_out_size(in_width,kernel_size[1],stride[1])
    ret_ = np_unfold(npx, kernel_size, stride).transpose(1,0,2).reshape(in_channel,*kernel_size,batch_size,*tmp0).copy()

    ms0 = ms.Tensor(npx, dtype=ms.float32)
    img2col = ms.ops.operations.Im2Col(kernel_size=kernel_size, pad_mode='valid', stride=stride)
    ret0 = img2col(ms0).asnumpy() #3,7,7,32,112,112
    assert hfe(ret_, ret0) < 1e-4


def draft_npu_img2col():
    # # test_img2col
    batch_size=32
    in_channel=3
    in_height=224
    in_width=224
    kernel_size=(7,7)
    stride=(2,2)

    npx = np.random.randn(batch_size, in_channel, in_height, in_width).astype(np.float32)

    hf_out_size = lambda in_size,kernel_size,stride: 1 + (in_size-(kernel_size-1)-1) // stride
    tmp0 = hf_out_size(in_height,kernel_size[0],stride[0]), hf_out_size(in_width,kernel_size[1],stride[1])
    ret_ = np_unfold(npx, kernel_size, stride).transpose(1,0,2).reshape(in_channel,*kernel_size,batch_size,*tmp0).copy()
    # (3, 7, 7, 32, 109, 109)

    ms0 = ms.Tensor(npx, dtype=ms.float16)
    op0 = ms.ops.operations.CusImg2Col(ksizes=(1,*kernel_size,1), strides=(1,*stride,1))
    op1 = ms.ops.operations.CusMatMulCube(transpose_a=True)
    tmp0 = op0(ms0) #(401408, 784)
    ret0 = op1(tmp0, tmp0).asnumpy() #(784,784)


def test_CusMatrixCombine():
    for N0 in [1,2,4,8,16]:
        np0 = np.random.randn(N0,128,128).astype(np.float32)
        ret_ = np.zeros((N0,128,N0,128), dtype=np0.dtype)
        ret_[range(N0),:,range(N0)] = np0
        ret_ = ret_.reshape(N0*128, N0*128)

        ms0 = ms.Tensor(np0, ms.float32)
        op = ms.ops.operations.CusMatrixCombine()
        ret0 = op(ms0).asnumpy()
        assert hfe(ret_,ret0) < 1e-6, N0


def my_guess_CusFusedAbsMax1(np0, N0=16):
    assert (np0.ndim==3) and (np0.shape[1]==np0.shape[2]) and (np0.shape[1]%N0==0)
    N1 = np0.shape[1]//N0
    tmp0 = np.diagonal(np0, axis1=1, axis2=2)
    ret = tmp0.reshape(np0.shape[0]*N1, N0).max(axis=1)
    return ret

def my_guess_CusFusedAbsMax1(np0, N0=16):
    assert (np0.ndim==3) and (np0.shape[1]==np0.shape[2]) and (np0.shape[1]%N0==0)
    N1 = np0.shape[1]//N0
    tmp0 = np.diagonal(np0.reshape(np0.shape[0],N1,N0,N1,N0), axis1=1, axis2=3)
    ret = tmp0.max(axis=(1,2)).reshape(-1)
    return ret

N0 = 4
N1 = 8
np0 = np.random.randn(N0, N1*16, N1*16).astype(np.float32)
np0 = np0 + np0.transpose(0,2,1)
ret_ = my_guess_CusFusedAbsMax1(np0)
ms0 = ms.Tensor(np0)
op = ms.ops.operations.CusFusedAbsMax1()
ret0 = op(ms0).asnumpy()
assert np.std(ret0, axis=1).max() < 1e-6
ret0 = ret0[:,0]
hfe(ret_, ret0)

hf0 = lambda *x: [tuple(int(z) for z in y) for y in zip(*x)]
[hf0(*np.where(np.abs(np0-x)<5e-6)) for x in ret0]
np.sort(np.abs(zc0.reshape(-1) - zc1[0]))[:100]


# def test_CusFusedAbsMax1
hf_definite = lambda x: np.matmul(x, x.T)
N0 = random.choice([1,2,4,8,16]) #fail for N0=3,5,6, large error for N0=1,2
split_dim = 128
np0 = hf_definite(np.random.randn(N0*split_dim, N0*split_dim).astype(np.float32))

op0 = ms.ops.operations.CusCholeskyTrsm() #split_dim must be 128
op1 = ms.ops.operations.CusBatchMatMul()
op2 = ms.ops.operations.CusFusedAbsMax1()
ms0 = ms.Tensor(np0, dtype=ms.float32)
tmp0 = op0(ms0)
tmp1 = op1(tmp0, tmp0) #(4,128,128)
z0 = op2(tmp1) #(ms,float32,(32,64))
z1 = op2(z0) #(ms,float32,(1,))
