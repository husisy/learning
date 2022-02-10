import numpy as np
import torch
import torch.nn.functional as F

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


def np_cross_entropy(logits, target):
    tmp0 = np.exp(logits - logits.max(axis=1, keepdims=True))
    tmp1 = np.log(tmp0 / tmp0.sum(axis=1, keepdims=True))
    ret = - np.mean(tmp1[np.arange(tmp0.shape[0]), target])
    return ret

def np_binary_cross_entropy(probability, target):
    ret = np.zeros_like(probability)
    ind0 = target.astype(np.bool)
    ret[ind0] = -np.log(probability[ind0])
    ind1 = np.logical_not(ind0)
    ret[ind1] = -np.log(1-probability[ind1])
    ret = ret.mean()
    return ret

def np_binary_cross_entropy_with_logits(logits, target):
    probability = 1/(1+np.exp(-logits))
    ret = np_binary_cross_entropy(probability, target)
    return ret

def test_F_cross_entropy(N0=3, N1=5):
    np0 = np.random.randn(N0, N1).astype(np.float32)
    np1 = np.random.randint(N1, size=(N0,))
    ret_ = np_cross_entropy(np0, np1)
    ret0 = F.cross_entropy(torch.tensor(np0), torch.tensor(np1, dtype=torch.int64))
    assert hfe(ret_, ret0.item()) < 1e-5


def test_nn_LogSoftmax_NLLLoss(N0=3, N1=5, N2=7):
    np0 = np.random.rand(N0, N1, N2).astype(np.float32)
    np1 = np.random.randint(N1, size=(N0,N2))
    ret_ = np_cross_entropy(np0.transpose(0,2,1).reshape(N0*N2,N1), np1.reshape(-1))

    hf_softmax = torch.nn.LogSoftmax(dim=1)
    hf_loss = torch.nn.NLLLoss()
    ret0 = hf_loss(hf_softmax(torch.tensor(np0)), torch.tensor(np1, dtype=torch.int64))
    assert hfe(ret_, ret0.item()) < 1e-5


def test_nn_CrossEntropyLoss(N0=3, N1=5):
    np0 = np.random.randn(N0, N1).astype(np.float32)
    np1 = np.random.randint(N1, size=(N0,))
    ret_ = F.cross_entropy(torch.tensor(np0), torch.tensor(np1, dtype=torch.int64))
    hf_loss = torch.nn.CrossEntropyLoss()
    ret0 = hf_loss(torch.tensor(np0), torch.tensor(np1,dtype=torch.int64))
    assert hfe(ret_.item(), ret0.item()) < 1e-7


def test_nn_BCELoss(N0=13):
    # NOT prefered
    np0 = np.random.rand(N0).astype(np.float32)
    np1 = np.random.randint(2, size=(N0,))
    ret_ = np_binary_cross_entropy(np0, np1)
    ret0 = torch.nn.BCELoss()(torch.Tensor(np0), torch.Tensor(np1)).numpy()
    assert hfe(ret_, ret0) < 1e-5
    ret1 = F.binary_cross_entropy(torch.Tensor(np0), torch.Tensor(np1)).numpy()
    assert hfe(ret_, ret1) < 1e-5


def test_F_binary_cross_entropy_with_logits(N0=13):
    # use F.binary_cross_entropy_with_logits(), do NOT use F.binary_cross_entropy()
    np0 = np.random.randn(N0).astype(np.float32)
    np1 = np.random.randint(2, size=(N0,))
    ret_ = np_binary_cross_entropy_with_logits(np0, np1)
    ret0 = F.binary_cross_entropy_with_logits(torch.Tensor(np0), torch.Tensor(np1)).numpy()
    assert hfe(ret_, ret0) < 1e-5
