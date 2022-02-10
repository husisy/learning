import numpy as np

import sklearn
import sklearn.neighbors

def _my_naive_nn(np0):
    if len(np0) > 1e3:
        return None,None
    ret_index = []
    ret_value = []
    for ind0 in range(len(np0)):
        tmp0 = np.sum((np0[ind0]-np0)**2, axis=1)
        tmp0[ind0] = np.inf
        ind1 = np.argmin(tmp0)
        ret_index.append(ind1)
        ret_value.append(tmp0[ind1])
    return np.array(ret_index, dtype=np.int64), np.array(ret_value, dtype=np.float64)

def test_neighbors_NearestNeighbors(N0=100):
    np0 = np.random.rand(N0, 3)
    ret0_,ret1_ = _my_naive_nn(np0)
    tmp0 = sklearn.neighbors.NearestNeighbors(algorithm='auto', n_neighbors=1)
    tmp0.fit(np0)
    tmp1 = tmp0.kneighbors_graph(mode='distance')
    # NearestNeighbors on 10000 points takes up around 0.05 second
    ret0 = tmp1.indices
    ret1 = tmp1.data**2
    assert np.all(ret0_==ret0)
    assert np.abs(ret1_-ret1).max() < 1e-5
