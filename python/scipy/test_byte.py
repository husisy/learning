import os
import numpy as np

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())

def test_write_read_byte(N0=233):
    np0 = np.random.randint(0, 256, size=(233,), dtype=np.uint8)
    filepath = hf_file('tbd00.byte')
    np0.tofile(filepath)

    with open(filepath, 'rb') as fid:
        z0 = np.frombuffer(fid.read(), dtype=np.uint8)
    z1 = np.fromfile(filepath, dtype=np.uint8)
    assert np.all(np0==z0)
    assert np.all(np0==z1)
