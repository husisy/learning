import numpy as np
import torch

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hf_out_size = lambda in_size,kernel_size,stride: 1 + (in_size-(kernel_size-1)-1) // stride


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


def test_torch_unfold(batch_size=2, in_channel=3, out_channel=2,
            in_height=57, in_width=59, kernel_size=(4,5), stride=(2,3), padding=(1,2)):
    out_height = hf_out_size(in_height+2*padding[0], kernel_size[0], stride[0])
    out_width = hf_out_size(in_width+2*padding[1], kernel_size[1], stride[1])
    npx = np.random.randn(batch_size, in_channel, in_height, in_width)
    torch_x = torch.tensor(npx, dtype=torch.float64)

    ret_ = []
    npx_padded = np.pad(npx, [(0,0),(0,0)] + [(x,x) for x in padding])
    for ind0 in range(0, out_height*stride[0], stride[0]):
        for ind1 in range(0, out_width*stride[1], stride[1]):
            tmp0 = npx_padded[:,:,ind0:(ind0+kernel_size[0]),ind1:(ind1+kernel_size[1])]
            ret_.append(tmp0.reshape(batch_size, -1))
    ret_ = np.stack(ret_, axis=2) #(batch_size, in_channel*kernel_size, ?)

    ret0 = torch.nn.functional.unfold(torch_x, kernel_size, stride=stride, padding=padding).numpy()
    ret1 = np_unfold(npx, kernel_size, stride, [(x,x) for x in padding])
    assert hfe(ret_, ret0) < 1e-7
    assert hfe(ret_, ret1) < 1e-7


def test_torch_nn_conv1d(batch_size=2, in_channel=3, out_channel=2,
            in_height=57, kernel_size=4, stride=3, padding=2):
    out_height = hf_out_size(in_height+2*padding, kernel_size, stride)
    npx = np.random.randn(batch_size, in_channel, in_height)
    npw = np.random.randn(out_channel, in_channel, kernel_size)
    torch_x = torch.tensor(npx, dtype=torch.float64)
    torch_w = torch.tensor(npw, dtype=torch.float64)

    tmp0 = np_unfold(npx, [kernel_size], [stride], [(padding,padding)])
    tmp1 = tmp0.transpose(0,2,1).reshape(-1,tmp0.shape[1]) @ npw.reshape(out_channel,-1).transpose(1,0)
    ret_ = tmp1.reshape(batch_size, out_height, out_channel).transpose(0,2,1)

    ret0 = torch.nn.functional.conv1d(torch_x, torch_w, stride=stride, padding=padding).numpy()
    assert hfe(ret_, ret0) < 1e-7


def test_torch_nn_conv2d(batch_size=2, in_channel=3, out_channel=2,
            in_height=57, in_width=59, kernel_size=(4,5), stride=(2,3), padding=(1,2)):
    out_height = hf_out_size(in_height+2*padding[0], kernel_size[0], stride[0])
    out_width = hf_out_size(in_width+2*padding[1], kernel_size[1], stride[1])
    npx = np.random.randn(batch_size, in_channel, in_height, in_width)
    npw = np.random.randn(out_channel, in_channel, *kernel_size)
    torch_x = torch.tensor(npx, dtype=torch.float64)
    torch_w = torch.tensor(npw, dtype=torch.float64)

    tmp0 = np_unfold(npx, kernel_size, stride, [(x,x) for x in padding])
    tmp1 = tmp0.transpose(0,2,1).reshape(-1,tmp0.shape[1]) @ npw.reshape(out_channel,-1).transpose(1,0)
    ret_ = tmp1.reshape(batch_size, out_height, out_width, out_channel).transpose(0,3,1,2)

    ret0 = torch.nn.functional.conv2d(torch_x, torch_w, stride=stride, padding=padding).numpy()
    assert hfe(ret_, ret0) < 1e-7
