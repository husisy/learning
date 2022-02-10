import numpy as np
from collections import Counter
import torch

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

def test_distributions_Categorical(N0=3, N1=5, N2=100000):
    tmp0 = np.random.rand(N0,N1)
    np0 = tmp0 / tmp0.sum(axis=1, keepdims=True)

    torch0 = torch.tensor(np0, dtype=torch.float64)
    torch1 = torch.distributions.Categorical(torch0).sample([N2])
    tmp0 = (Counter(x.tolist()) for x in torch1.numpy().T)
    ret0 = np.array([[x[y] for y in range(N1)] for x in tmp0])/N2

    assert hfe(np0, ret0) < 0.1, 'relative error should be small'
