import os
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

hf_file = lambda *x: os.path.join('tbd00', *x)

with open(hf_file('tbd00.txt'), 'r', encoding='utf-8') as fid:
    tmp0 = [x.strip() for x in fid]
    tmp0 = [x for x in tmp0 if x]
    assert {y for x in tmp0 for y in x} <= {'0','1'}
    z0 = np.array([[y=='1' for y in x] for x in tmp0])

fig,ax = plt.subplots()
ax.imshow(z0)
