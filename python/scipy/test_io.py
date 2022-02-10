import os
import numpy as np
import scipy.io

hf_file = lambda *x: os.path.join('tbd00', *x)
hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

if not os.path.exists(hf_file()):
    os.makedirs(hf_file())


def test_interact_with_matlab_data():
    filepath = hf_file('tbd00.mat')

    np0 = np.random.randn(3,4)
    np1 = np.random.randn(4,3)
    scipy.io.savemat(filepath, {'np0':np0, 'np1':np1})
    # scipy.io.whosmat(filepath)
    z0 = scipy.io.loadmat(filepath)
    assert hfe(np0, z0['np0']) < 1e-7
    assert hfe(np1, z0['np1']) < 1e-7

    # struct
    np0 = {
        'field0': np.random.randn(3,4),
        'field1': np.random.randn(4,3),
    }
    scipy.io.savemat(filepath, {'np0':np0})
    z0 = scipy.io.loadmat(filepath, struct_as_record=False)
    tmp0 = z0['np0'][0,0] #struct must be 2d for MATLAB
    assert hfe(np0['field0'], tmp0.field0) < 1e-7
    assert hfe(np0['field1'], tmp0.field1) < 1e-7
    # TODO cell
