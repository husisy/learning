import numpy as np
import torch

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


def test_torch_multinomial():
    np0 = np.random.rand(5)
    np0 = np0 / np0.sum()
    num_sample = 100000
    tmp0 = torch.multinomial(torch.tensor(np0), num_sample, replacement=True).numpy()
    tmp2 = np.unique(tmp0, return_counts=True)[1]/num_sample
    assert hfe(np0, tmp2) < 0.01
    # TODO how to unittest replacement=False


def test_torch_cupy_share_data():
    import torch.utils.dlpack
    import cupy as cp
    np0 = np.random.rand(3, 5)
    torch0 = torch.tensor(np0.copy()).to('cuda')

    cp0 = cp.fromDlpack(torch.utils.dlpack.to_dlpack(torch0))
    cp0[0,0] = 0.233
    assert hfe(torch0.to('cpu').numpy(), cp0.get()) < 1e-5

    cp1 = cp.array(np0)
    torch1 = torch.utils.dlpack.from_dlpack(cp1.toDlpack())
    torch1[0,0] = 0.233
    assert hfe(torch1.to('cpu').numpy(), cp1.get()) < 1e-5


def test_torch_memory_format():
    hf_shape_to_stride = lambda x: tuple(np.cumprod(np.asarray(x[1:])[::-1])[::-1].tolist() + [1])
    shape = (3,4,5,6)
    stride0 = hf_shape_to_stride(shape)
    tmp0 = hf_shape_to_stride((shape[0],shape[2],shape[3],shape[1]))
    stride1 = tmp0[0],tmp0[3],tmp0[1],tmp0[2]
    torch0 = torch.randn(*shape) #default torch.contiguous_format
    assert (torch0.stride()==stride0) and (tuple(torch0.shape)==shape)
    assert torch0.is_contiguous(memory_format=torch.contiguous_format)

    torch1 = torch0.contiguous(memory_format=torch.channels_last)
    assert (torch1.stride()==stride1) and (tuple(torch1.shape)==shape)
    assert torch1.is_contiguous(memory_format=torch.channels_last)

    torch2 = torch1.contiguous(memory_format=torch.contiguous_format)
    assert (torch2.stride()==stride0) and (tuple(torch2.shape)==shape)
    assert torch2.is_contiguous(memory_format=torch.contiguous_format)


def test_nn_utils_rnn_pad_sequence():
    # de_batch = [torch.tensor([BOS_IDX]+x+[EOS_IDX], dtype=torch.int64) for x,_ in data_batch]
    # de_batch = torch.nn.utils.rnn.pad_sequence(de_batch, padding_value=PAD_IDX)
    # en_batch = [torch.tensor([BOS_IDX]+x+[EOS_IDX], dtype=torch.int64) for _,x in data_batch]
    # en_batch = torch.nn.utils.rnn.pad_sequence(en_batch, padding_value=PAD_IDX)
    PAD_IDX = 233
    N0 = 3
    np_rng = np.random.default_rng()
    np0 = [np_rng.integers(0, int(1e4), size=np_rng.integers(5,10)) for _ in range(N0)]

    tmp0 = max(len(x) for x in np0)
    tmp1 = [np.pad(x, [(0,tmp0-len(x))],mode='constant', constant_values=PAD_IDX) for x in np0]
    ret0_ = np.stack(tmp1, axis=1)
    ret1_ = np.stack(tmp1, axis=0)
    torch0 = [torch.tensor(x, dtype=torch.int32) for x in np0]
    ret0 = torch.nn.utils.rnn.pad_sequence(torch0, padding_value=PAD_IDX, batch_first=False).numpy() #batch_first=False(default)
    ret1 = torch.nn.utils.rnn.pad_sequence(torch0, padding_value=PAD_IDX, batch_first=True).numpy()
    assert hfe(ret0_, ret0) < 1e-5
    assert hfe(ret1_, ret1) < 1e-5


def test_strange_masked_fill_in_multiHeadAttention(N0=3, N1=5, N2=7):
    np_rng = np.random.default_rng()
    np0 = np_rng.uniform(0, 1, size=(N0, 1, N2)) > 0.5
    np1 = np_rng.normal(size=(N1,N2)).astype(np.float32)

    ret_ = np.tile(np1[np.newaxis], (N0,1,1))
    ret_[np.tile(np0, (1,N1,1))] = float('-inf')

    torch0 = torch.tensor(np0, dtype=torch.bool)
    torch1 = torch.tensor(np1, dtype=torch.float32)
    ret0 = torch1.masked_fill(torch0, float('-inf')).numpy()

    assert np.all(np.isneginf(ret_)==np.isneginf(ret0))
    tmp0 = ret_.copy()
    tmp0[np.isneginf(tmp0)] = 0
    tmp1 = ret0.copy()
    tmp1[np.isneginf(tmp1)] = 0
    assert hfe(tmp0, tmp1) < 1e-5


def test_softplus(N0=13):
    np_rng = np.random.default_rng()
    beta = np_rng.uniform(0, 1)
    np0 = np_rng.uniform(-2, 2, size=(N0,)).astype(np.float32)

    torch0 = torch.tensor(np0)
    ret_ = np.log(1 + np.exp(beta*np0)) / beta
    ret0 = torch.nn.functional.softplus(torch0, beta).numpy()
    assert hfe(ret_, ret0) < 1e-5
