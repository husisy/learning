import numpy as np
import mindspore as ms
import te
import topi
import te.tik

from demo_npu03_laji_kernel import CusMatrixCombine #error if put in one file

ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="Ascend")
hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

class CusMatrixCombine(ms.ops.PrimitiveWithInfer):

    @ms.ops.prim_attr_register
    def __init__(self):
        self.CusMatrixCombine = CusMatrixCombine
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def infer_shape(self, data_shape):
        a, b, c = data_shape
        shape = [a * b, a * c]

        return shape

    def infer_dtype(self, data_dtype):
        return data_dtype


if __name__=='__main__':
    N0 = 4
    np0 = np.random.randn(N0,128,128).astype(np.float32)
    ret_ = np.zeros((N0,128,N0,128), dtype=np0.dtype)
    ret_[range(N0),:,range(N0)] = np0
    ret_ = ret_.reshape(N0*128, N0*128)

    ms0 = ms.Tensor(np0, ms.float32)
    op = CusMatrixCombine()
    ret0 = op(ms0).asnumpy()
    assert hfe(ret_,ret0) < 1e-6
