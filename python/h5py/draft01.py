import io
import h5py
import PIL.Image
import numpy as np


def demo_hdf5_image00():
    # https://github.com/h5py/h5py/issues/745#issuecomment-259646466
    # https://github.com/h5py/h5py/issues/745#issuecomment-339451940
    filepath = '/path/to/xxx.JPEG'
    ret_ = np.asarray(PIL.Image.open(filepath))
    with h5py.File('tbd00.hdf5', 'w') as fid:
        dt = h5py.special_dtype(vlen=np.dtype('uint8'))
        dataset0 = fid.create_dataset('dataset0', (1,), dtype=dt)
        with open(filepath, 'rb') as image_fid:
            dataset0[0] = np.frombuffer(image_fid.read(), dtype='uint8')
        print(dataset0.shape) #(1,)
        print(dataset0.dtype) #object

    with h5py.File('tbd00.hdf5', 'r') as fid:
        ret0 = np.asarray(PIL.Image.open(io.BytesIO(fid['dataset0'][0])))
    assert np.all(ret_==ret0)


def demo_hdf5_image01():
    # https://github.com/h5py/h5py/issues/745#issuecomment-521819041
    filepath = '/path/to/xxx.JPEG'
    ret_ = np.asarray(PIL.Image.open(filepath))
    with h5py.File('tbd00.hdf5', 'w') as fid:
        with open(filepath, 'rb') as image_fid:
            dataset0 = fid.create_dataset('dataset0', data=np.asarray(image_fid.read()))
            print(dataset0.shape) #()
            print(dataset0.dtype) #|S165934
    with h5py.File('tbd00.hdf5', 'r') as fid:
        # tmp0 = np.asarray(fid['dataset0'])
        tmp0 = fid['dataset0'][...]
        ret0 = np.asarray(PIL.Image.open(io.BytesIO(tmp0)))
    assert np.all(ret_==ret0)
