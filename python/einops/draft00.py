import einops
import numpy as np

hfe = lambda x, y, eps=1e-5: np.max(np.abs(x - y) / (np.abs(x) + np.abs(y) + eps))
np_rng = np.random.default_rng()

np0 = np_rng.uniform(size=(6,96,96,3))
ret0_ = np0.transpose(0, 2, 1, 3)
ret0 = einops.rearrange(np0, 'b h w c -> b w h c')
assert hfe(ret0_, ret0) < 1e-7

ret0 = einops.rearrange(np0, 'b h w c -> (b h) w c')
ret0_ = np0.reshape(np0.shape[0]*np0.shape[1], np0.shape[2], np0.shape[3])
assert hfe(ret0_, ret0) < 1e-7

ret0 = einops.rearrange(np0, 'b h w c -> h (b w) c')
ret0_ = np0.transpose(1,0,2,3).reshape(np0.shape[1], np0.shape[0]*np0.shape[2], np0.shape[3])
assert hfe(ret0_, ret0) < 1e-7

ret0 = einops.rearrange(np0, '(b1 b2) h w c -> b1 b2 h w c ', b1=2)
ret0_ = np0.reshape(2, 3, *np0.shape[1:])
assert hfe(ret0_, ret0) < 1e-7

## reduce
ret0 = einops.reduce(np0, 'b h w c -> h w c', 'mean')
ret0_ = np0.mean(axis=0)
assert hfe(ret0_, ret0) < 1e-7

ret0 = einops.reduce(np0, 'b h w c -> h w', 'min')
ret0_ = np0.min(axis=(0,3))
assert hfe(ret0_, ret0) < 1e-7

ret0 = einops.reduce(np0, 'b (h h2) (w w2) c -> h (b w) c', 'mean', h2=2, w2=2)
ret0_ = np0.reshape(6,48,2,48,2,3).mean(axis=(2,4)).transpose(1,0,2,3).reshape(48,-1,3)
assert hfe(ret0_, ret0) < 1e-7
