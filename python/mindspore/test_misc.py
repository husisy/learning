import numpy as np
import mindspore as ms

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="GPU")
# ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="GPU")


def test_second_order_derivative():
    grad_all = ms.ops.GradOperation()
    # func = lambda x: x**3 #fail
    def func(x):
        return x**2.33

    def df_func(x):
        return grad_all(func)(x)

    @ms.ms_function
    def df2_func(x):
        return grad_all(df_func)(x)
    np0 = np.random.rand(3).astype(np.float32)
    ms0 = ms.Tensor(np0)
    ms1 = df_func(ms0)
    assert hfe(2.33*np0**1.33, ms1.asnumpy()) < 1e-5
    ms2 = df2_func(ms0)
    assert hfe(1.33*2.33*np0**0.33, ms2.asnumpy()) < 1e-5


class DummyNet00(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv = ms.nn.Conv2d(3, 64, 3)
    def construct(self, x):
        x = self.conv(x)
        return x

def test_Cell_structure():
    net = DummyNet00()
    z0 = {k:v for k,v in net.cells_and_names()} #exclude ms.ops.Conv2d
    assert id(z0[''])==id(net)
    assert id(z0['conv'])==id(net.conv)


def clip_gradient(dx):
    ret = ms.ops.clip_by_value(dx, -1, 1)
    return ret

class DummyNet03(ms.nn.Cell):
    def __init__(self, op):
        super().__init__()
        self.op = op
    def construct(self, x, y):
        x = self.op(x)
        y = self.op(y)
        ret = x*y
        return ret

def test_InsertGradientOf():
    np0 = np.random.randn(10).astype(np.float32)
    np1 = np.random.randn(10).astype(np.float32)
    ret0_ = np0 * np1
    ret1_ = np.clip(np1, -1, 1)
    ret2_ = np.clip(np0, -1, 1)

    clip = ms.ops.InsertGradientOf(clip_gradient)

    ms0 = ms.Tensor(np0, dtype=ms.float32)
    ms1 = ms.Tensor(np1, dtype=ms.float32)
    net = DummyNet03(clip)
    ret0 = net(ms0, ms1).asnumpy()
    ret1,ret2 = ms.ops.GradOperation(get_all=True)(net)(ms0, ms1)
    ret1 = ret1.asnumpy()
    ret2 = ret2.asnumpy()
    assert hfe(ret0_, ret0) < 1e-5
    assert hfe(ret1_, ret1) < 1e-5
    assert hfe(ret2_, ret2) < 1e-5



# def test_matrix_diag():
#     # ascend only
#     np0 = np.random.randn(3).astype(np.float32)
#     ms0 = ms.Tensor(np0)
#     op = ms.nn.MatrixDiag()
#     ms1 = op(ms0)

#     np0 = np.random.randn(3, 5, 5).astype(np.float32)
#     np1 = np.random.randn(3, 5).astype(np.float32)
#     ms0 = ms.Tensor(np0)
#     ms1 = ms.Tensor(np1)
#     op = ms.nn.MatrixSetDiag()
#     ms2 = op(ms0, ms1)


def test_RandomChoiceWithMask(N0=13, N1=5, N2=9):
    for _ in range(100):
        np0 = np.random.rand(N0,N1)>0.5
        if np0.sum()>=N2:
            # when np0.sum() < N2, the behavior of RandomChoiceWithMask is bad defined
            break
    op = ms.ops.RandomChoiceWithMask(count=N2)
    ms0 = ms.Tensor(np0)
    ms1, ms1_mask = op(ms0)
    assert np.all(ms1_mask.asnumpy())
    tmp0 = ms1.asnumpy()
    assert np.all(np0[tmp0[:,0], tmp0[:,1]])
    assert len(set(tuple(x) for x in tmp0.tolist())) == N2


class DummyNet04(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.op_uniform = ms.ops.UniformReal()
    def construct(self, x):
        # ret = np.random.rand(1).astype(np.float32) * x #fail
        # ret = random.random() * x #fail
        ret = self.op_uniform((1,))*x
        return ret

def test_class_uniformreal():
    ms0 = ms.Tensor(np.random.rand(3), dtype=ms.float32)
    net = DummyNet04()
    z0 = [net(ms0) for _ in range(5)] #only the last one is kept
    z0 = [x.asnumpy() for x in z0]
    assert all(hfe(x,z0[0])<1e-5 for x in z0[1:])
    net = DummyNet04()
    z1 = [net(ms0).asnumpy() for _ in range(5)]
    assert all(hfe(x,z1[0])>1e-3 for x in z1[1:])
