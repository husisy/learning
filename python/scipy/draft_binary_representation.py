import numpy as np

np_i_type = {np.int8,np.int16,np.int32,np.int64}
np_f_type = {np.float16,np.float32,np.float64}
np_ui_type = {np.uint8,np.uint16,np.uint32,np.uint64}
np_i_to_ui = {np.int8:np.uint8, np.int16:np.uint16, np.int32:np.uint32, np.int64:np.uint64}
np_f_to_ui = {np.float16:np.uint16, np.float32:np.uint32, np.float64:np.uint64}

def hfb(np0):
    np0 = np.asarray(np0)
    ndim = np0.ndim
    assert ndim in {0,1}
    nptype = np0.dtype.type
    if nptype in np_i_to_ui:
        np0 = np0.view(np_i_to_ui[nptype])
    if nptype in np_f_to_ui:
        np0 = np0.view(np_f_to_ui[nptype])
    nptype = np0.dtype.type
    assert nptype in np_ui_type
    ret = [np.binary_repr(x, width=np.asarray(x).itemsize * 8) for x in np0.reshape(-1)]
    if ndim==0:
        ret = ret[0]
    return ret


def demo_float16():
    np0 = np.zeros(1, dtype=np.float16)
    np1 = np.nextafter(np0, np0+1)
    np2 = np.nextafter(np0, np0-1)
    print('float16(+delta)', np1.item(), hfb(np1))
    print('float16(-delta)', np2.item(), hfb(np2))

    np0 = np.ones(1, dtype=np.float16)
    np1 = np.nextafter(np0, np0+1)
    np2 = np.nextafter(np0, np0-1)
    print('float16(1+delta)', np1.item(), hfb(np1))
    print('float16(1-delta)', np2.item(), hfb(np2))


def array_byte_ordering():
    np0 = np.array([0, 3, 7, 15], dtype=np.uint8)
    np1 = np0.view(np.uint16)
    print(hfb(np0), hfb(np1))

    np0 = np.random.randint(2**15, size=(2,)).astype(np.uint16)
    np1 = np0.view(np.uint8)
    print(hfb(np0), hfb(np1))
