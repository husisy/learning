import numpy as np

# <U5: little-endian unicode string data, with a maximum length of 5 unicode code points
np0 = np.array(["hello", "world"]) #dtype='<U5'

# |S5: 5-byte string data
np1 = np.array([b"hello", b"world"])

# |V5: 5-byte void data
np2 = np1.astype(np.void)

## numpy-v2.0 required
# np.dtypes.StringDType
