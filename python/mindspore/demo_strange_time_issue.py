import numpy as np
import scipy.linalg
import mindspore as ms
import scipy.sparse
# import scipy.sparse.linalg
import pytest

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target='Ascend')

def generate_hermite_matrix(N0, min_eig=1, max_eig=2):
    ret = np.random.randn(N0,N0) + 1j*np.random.randn(N0,N0)
    ret = ret + np.conjugate(ret.T)
    eig0 = scipy.sparse.linalg.eigs(ret, k=1, which='SR', return_eigenvectors=False)
    eig1 = scipy.sparse.linalg.eigs(ret, k=1, which='LR', return_eigenvectors=False)
    ret = (ret - eig0*np.eye(N0)) * (max_eig-min_eig)/(eig1-eig0) + min_eig*np.eye(N0)
    return ret

%timeit generate_hermite_matrix(1024, 1, 2)
