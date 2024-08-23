import numpy as np
import scipy.stats

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))


def test_stats_entropy():
    N0 = 23
    N1 = 10
    hf_norm = lambda x: x/x.sum(axis=-1, keepdims=True)
    np0 = hf_norm(np.random.rand(N1))
    np1 = hf_norm(np.random.rand(N1))
    ret_ = scipy.stats.entropy(np0, np1)
    ret0 = np.sum(np0*np.log(np0/np1))
    abs(ret_-ret0) < 1e-10

    np0 = hf_norm(np.random.rand(N0,N1))
    np1 = hf_norm(np.random.rand(N0, N1))
    ret_ = scipy.stats.entropy(np0, np1, axis=1)
    ret0 = np.sum(np0*np.log(np0/np1), axis=1)
    assert np.abs(ret_-ret0).max() < 1e-10


def test_stats_unitary_group():
    dim = 4
    np0 = scipy.stats.unitary_group.rvs(dim)
    assert np.abs(np0 @ np0.T.conj() - np.eye(dim)).max() < 1e-10
    assert abs(abs(np.linalg.det(np0))-1) < 1e-10 #unitary but not special unitary
