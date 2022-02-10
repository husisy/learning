import numpy as np

from zctest_pybind11._cpp import CArray, c_sum, cuda_plus, cuda_plus_return_np

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hfe_r5 = lambda x,y,eps=1e-5: round(hfe(x,y,eps),5)

# CArray is 1-d double dtype only

def test_CArray_numpy_view():
    cnp0 = CArray(3)
    np0 = cnp0.numpy()
    np0[1] = 2.33 #all operation should be carried by CArray.numpy()
    tmp0 = hfe(cnp0.numpy(), np.array([0,2.33,2]))
    assert tmp0 < 1e-5, 'hfe={}'.format(tmp0)


def test_constructor():
    tmp0 = hfe(CArray(3).numpy(), np.arange(3))
    assert tmp0 < 1e-7, 'hfe={}'.format(tmp0)

    np0 = np.random.rand(3)
    cnp0 = CArray(np0)
    tmp0 = hfe(np0, cnp0.numpy())
    assert tmp0<1e-7, 'hfe={}'.format(tmp0)

def test_CArray_copy():
    # share memory
    cnp0 = CArray(3) #delete in cpp side
    _ = np.array(cnp0, copy=True) #delete in Python side

    # not share memory
    cnp1 = CArray(3) #delete in cpp side
    _ = np.array(cnp1, copy=False) #NOT delete in Python side

    # share memory
    np0 = np.random.rand(3) #delete in python side
    _ = CArray(np0, copy=True) #delete in cpp side

    # not share memory
    np1 = np.random.rand(3) #delete in python side
    _ = CArray(np1, copy=False) #NOT delete in cpp side


def test_CArray_get_info():
    np0 = np.random.rand(3)
    cnp0 = CArray(np0)
    print('np.random.rand(3): ', np0)
    print('Carry.get_info(): ', cnp0.get_info())

def test_stride_feature():
    # # np(strides!=size) -> CArray
    np0 = np.random.rand(3) + np.random.rand(3)*1j
    print('test_stride_feature:: hfe(real): ', hfe(np0.real, CArray(np0.real, copy=True)))
    print('test_stride_feature:: hfe(imag): ', hfe(np0.imag, CArray(np0.imag, copy=True)))

def test_c_sum():
    np0 = np.random.rand(233)
    tmp0 = hfe(np0.sum(), c_sum(np0))
    assert tmp0<1e-7, 'hfe={}'.format(tmp0)


def test_cuda_sum():
    num0 = 12
    np0 = np.random.rand(num0)
    np1 = np.random.rand(num0)
    cnp0 = cuda_plus(np0, np1) #if not use a variable, the .numpy() will be point to an invalid data
    tmp0 = hfe(np0+np1, cnp0.numpy())
    assert tmp0<1e-7, 'hfe={}'.format(tmp0)
    tmp0 = hfe(np0+np1, cuda_plus_return_np(np0, np1))
    assert tmp0<1e-7, 'hfe={}'.format(tmp0)
