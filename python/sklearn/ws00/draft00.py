import numpy as np


N0 = 10
hf_symmetry = lambda x: (x+x.T)/2
np_rng = np.random.default_rng()
# np0 = hf_symmetry(np_rng.normal(size=(N0,N0))) #fail
np0 = hf_symmetry(np_rng.uniform(size=(N0,N0)))
# np0 = hf_symmetry(np_rng.uniform(size=(N0,N0))>0.5).astype(np.float32)
np1 = np.diag(np0.sum(axis=1)) - np0
print(np.linalg.eigvalsh(np1)[:3])

N0 = 10
N1 = 5
np_rng = np.random.default_rng()
tmp0 = np_rng.uniform(size=(N0, N1))
np0 = tmp0 / tmp0.sum(axis=1, keepdims=True)
np1 = np0 @ np.diag(1/(np0.sum(axis=0))) @ np0.T
np.linalg.eigvalsh(np1)


N0 = 10
N1 = 5
np0 = np_rng.normal(size=(N0,N1))
z0 = np.linalg.eigvalsh(np0 @ np0.T)
z1 = np.linalg.eigvalsh(np0.T @ np0)
print(z0)
print(z1)
