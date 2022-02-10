import random
import numpy as np
import scipy.linalg
import mindspore as ms
import scipy.sparse
import scipy.sparse.linalg
import pytest

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

from utils import detect_target_device

ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target=detect_target_device())


def test_advanced_indexing0():
    np0 = np.random.rand(3,4,5).astype(np.float32)
    ind0 = [1,3]
    ret_ = np0[:,ind0]

    ms0 = ms.Tensor(np0)
    ind0_ms = ms.Tensor(ind0, dtype=ms.int32)
    ret0 = ms.ops.gather(ms0, ind0_ms, 1).asnumpy()
    assert hfe(ret_, ret0) < 1e-5
    # ret1 = ms0[:,ind0_ms].asnumpy() #mindspore-master-branch only
    # assert hfe(ret_,ret1) < 1e-5


def test_advanced_indexing1(N0=5, N1=7, N2=3):
    np0 = np.random.rand(N0,N1).astype(np.float32)
    np1 = np.random.randint(0, N0, size=(N2,)).astype(np.int32)
    np2 = np.random.randint(0, N1, size=(N2,)).astype(np.int32)
    ret_ = np0[np1,np2]
    ms0 = ms.Tensor(np0)
    ret0 = ms.ops.functional.gather_nd(ms0, ms.Tensor(np.stack([np1,np2],axis=1)))
    assert hfe(ret_, ret0.asnumpy()) < 1e-5
    # ret1 = ms0[ms.Tensor(np1), ms.Tensor(np2)] #ms-master only
    # assert hfe(ret_, ret1.asnumpy()) < 1e-5


def test_matmul():
    np0 = np.random.rand(3, 5).astype(np.float32)
    np1 = np.random.rand(5, 7).astype(np.float32)
    ret_ = np0 @ np1
    ret0 = ms.nn.MatMul()(ms.Tensor(np0), ms.Tensor(np1)).asnumpy()
    ret1 = ms.ops.MatMul()(ms.Tensor(np0), ms.Tensor(np1)).asnumpy()
    assert hfe(ret_, ret0) < 1e-5
    assert hfe(ret_, ret1) < 1e-5


def test_hyper_map():
    square = ms.ops.MultitypeFuncGraph('square')
    @square.register("Tensor")
    def square_tensor(x):
        return ms.ops.square(x)

    N0 = 5
    tmp0 = np.random.randint(3, 10, size=N0)
    np0 = [np.random.randn(x).astype(np.float32) for x in tmp0]
    ms0 = [ms.Tensor(x) for x in np0]

    ret_ = [x**2 for x in np0]

    hyper_map = ms.ops.HyperMap()
    z0 = hyper_map(square, ms0)
    ret0 = [x.asnumpy() for x in z0]
    assert all(hfe(x,y)<1e-5 for x,y in zip(ret_,ret0))

    map_square = ms.ops.HyperMap(square)
    z0 = map_square(ms0)
    ret0 = [x.asnumpy() for x in z0]
    assert all(hfe(x,y)<1e-5 for x,y in zip(ret_,ret0))

    map_op = ms.ops.composite.Map()
    z0 = map(square, ms0)
    ret0 = [x.asnumpy() for x in z0]
    assert all(hfe(x,y)<1e-5 for x,y in zip(ret_,ret0))

    map_square = ms.ops.composite.Map(square)
    z0 = map_square(ms0)
    ret0 = [x.asnumpy() for x in z0]
    assert all(hfe(x,y)<1e-5 for x,y in zip(ret_,ret0))


def test_misc():
    np0 = np.random.rand(3).astype(np.float32)
    np1 = np.random.rand(3).astype(np.float32)
    np2 = np.random.rand(1)[0].astype(np.float32)
    ret_ = np0*np2 + np1

    ms0 = ms.Tensor(np0)
    ms1 = ms.Tensor(np1)
    ms2 = ms.Tensor(np2)
    tmp0 = ms.ops.Mul()(ms0, ms2)
    ret0 = ms.ops.AddN()((tmp0, ms1)).asnumpy()
    ret1 = (ms0*ms2 + ms1).asnumpy()
    assert hfe(ret_,ret0) < 1e-5
    assert hfe(ret_, ret1) < 1e-5


def np_unfold(npx, kernel_size_list, stride_list=None, padding_list=None):
    N0 = npx.ndim
    assert (N0>2) and (len(kernel_size_list)==(N0-2))
    if stride_list is None:
        stride_list = (1,)*(N0-2)
    else:
        assert len(stride_list) == N0-2
    if padding_list is not None:
        assert (len(padding_list)==N0-2) and all(len(x)==2 for x in padding_list)
        npx = np.pad(npx, [(0,0),(0,0)] + padding_list)
    old_shape = npx.shape
    old_strides = npx.strides
    hf_out_size = lambda in_size,kernel_size,stride: 1 + (in_size-(kernel_size-1)-1) // stride
    out_size = [hf_out_size(x,y,z) for x,y,z in zip(old_shape[2:],kernel_size_list,stride_list)]
    new_shape = old_shape[:2] + tuple(y for x in zip(out_size,kernel_size_list) for y in x)
    tmp0 = [(x*y,x) for x,y in zip(old_strides[2:],stride_list)]
    new_strides = old_strides[:2] + tuple(y for x in tmp0 for y in x)
    ret = np.lib.stride_tricks.as_strided(npx, shape=new_shape, strides=new_strides).copy()
    assert not np.any(np.isnan(ret).reshape(-1))
    tmp0 = [0] + list(range(1,2*N0-2,2)) + list(range(2,2*N0-2,2))
    tmp1 = [old_shape[0], old_shape[1]*np.prod(kernel_size_list), np.prod(out_size)]
    ret233 = ret.transpose(*tmp0).reshape(tmp1)
    return ret233


@pytest.mark.skipif(ms.context.get_context('device_target')!='GPU', reason='GPU only')
def test_Im2Col(batch_size=2, in_channel=3, in_height=57, in_width=59, kernel_size=(4,5), stride=(2,3)):
    npx = np.random.randn(batch_size, in_channel, in_height, in_width).astype(np.float32)

    hf_out_size = lambda in_size,kernel_size,stride: 1 + (in_size-(kernel_size-1)-1) // stride
    tmp0 = hf_out_size(in_height,kernel_size[0],stride[0]), hf_out_size(in_width,kernel_size[1],stride[1])
    ret_ = np_unfold(npx, kernel_size, stride).transpose(1,0,2).reshape(in_channel,*kernel_size,batch_size,*tmp0).copy()

    ms0 = ms.Tensor(npx, dtype=ms.float32)
    img2col = ms.ops.operations.Im2Col(kernel_size=kernel_size, pad_mode='valid', stride=stride)
    ret0 = img2col(ms0).asnumpy() #3,7,7,32,112,112
    assert hfe(ret_, ret0) < 1e-4


class DummyNet00(ms.nn.Cell):
    def __init__(self, ms0_list):
        super().__init__()
        self.ms0_list = ms0_list

    def construct(self, x):
        # ret = ms.ops.addn([x*y for y in self.ms0_list]) #not support yet
        tmp0 = []
        for y in self.ms0_list:
            tmp0.append(x*y)
        ret = ms.ops.addn(tmp0)
        return ret

def test_list_comprehension():
    np0_list = [np.random.randn(3).astype(np.float32) for _ in range(5)]
    np1 = np.random.randn(3).astype(np.float32)
    ret_ = np.stack(np0_list).sum(axis=0)*np1

    ms0_list = [ms.Tensor(x) for x in np0_list]
    ms1 = ms.Tensor(np1)
    net = DummyNet00(ms0_list)
    ret0 = net(ms1).asnumpy()
    assert hfe(ret_, ret0) < 1e-5


def test_BatchMatMul():
    N0 = 3
    for N1 in [5,8,32,64]:
        np0 = np.random.rand(N0, N1, N1).astype(np.float32)
        np1 = np.random.rand(N0, N1, N1).astype(np.float32)
        np0T = np0.transpose((0,2,1))
        np1T = np1.transpose((0,2,1))
        ret0_ = np.matmul(np0, np1)
        ret1_ = np.matmul(np0, np1T)
        ret2_ = np.matmul(np0T, np1)
        ret3_ = np.matmul(np0T, np1T)
        # large difference for np.einsum-float16
        # ret0_ = np.einsum(np0, [0,1,2], np1, [0,2,3], [0,1,3], optimize=True)
        # ret1_ = np.einsum(np0, [0,1,2], np1, [0,3,2], [0,1,3], optimize=True)
        # ret2_ = np.einsum(np0, [0,2,1], np1, [0,2,3], [0,1,3], optimize=True)
        # ret3_ = np.einsum(np0, [0,2,1], np1, [0,3,2], [0,1,3], optimize=True)
        # print(f'N1={N1}:', hfe(ret0_, ret0), hfe(ret1_, ret1), hfe(ret2_, ret2), hfe(ret3_, ret3))

        ms0 = ms.Tensor(np0)
        ms1 = ms.Tensor(np1)
        op0 = ms.ops.BatchMatMul(transpose_a=False, transpose_b=False) #default
        op1 = ms.ops.BatchMatMul(transpose_a=False, transpose_b=True)
        op2 = ms.ops.BatchMatMul(transpose_a=True, transpose_b=False)
        op3 = ms.ops.BatchMatMul(transpose_a=True, transpose_b=True)
        ret0 = op0(ms0, ms1).asnumpy()
        ret1 = op1(ms0, ms1).asnumpy()
        ret2 = op2(ms0, ms1).asnumpy()
        ret3 = op3(ms0, ms1).asnumpy()
        # large difference for float32
        # print(f'N1={N1}:', hfe(ret0_, ret0), hfe(ret1_, ret1), hfe(ret2_, ret2), hfe(ret3_, ret3))
        assert hfe(ret0_, ret0) < 1e-3
        assert hfe(ret1_, ret1) < 1e-3
        assert hfe(ret2_, ret2) < 1e-3
        assert hfe(ret3_, ret3) < 1e-3


@pytest.mark.skipif(ms.context.get_context('device_target')!='GPU', reason='GPU only')
def test_Cholesky(N0=3, N1=5):
    # WARNING results different between ms-1.0.1 and ms-1.1.0
    if ms.__version__=='1.1.0':
        op = ms.ops.operations.CholeskyTrsm(split_dim=0)
    else:
        op = ms.ops.operations.Cholesky(split_dim=0)
    hfT_positive = lambda x: (x @ x.T)

    np0 = np.stack([hfT_positive(np.random.randn(N1,N1)) for _ in range(N0)], axis=0).astype(np.float32)
    ret_ = np.linalg.inv(np0)
    ms0 = ms.Tensor(np0, dtype=ms.float32)
    ms1 = op(ms0).asnumpy()
    ret0 = np.einsum(ms1, [0,2,1], ms1, [0,2,3], [0,1,3])
    assert hfe(ret_, ret0) < 1e-4 #sometimes fail for hfe()=1.4e-4


@pytest.mark.skipif(ms.context.get_context('device_target')!='GPU', reason='GPU only')
def test_CholeskyTrsm():
    hf_definite = lambda x: np.matmul(x, x.T)
    split_dim = 128
    N0 = np.random.randint(split_dim*2, split_dim*4)
    np0 = hf_definite(np.random.randn(N0,N0)).astype(np.float32)

    tmp0 = [x*split_dim for x in range(int(np.ceil(N0/split_dim))+1)]
    ret_ = [np.linalg.inv(np0[x:y,x:y]) for x,y in zip(tmp0[:-1],tmp0[1:])]
    if ret_[-1].shape[0]<split_dim:
        ret_[-1] = scipy.linalg.block_diag(ret_[-1], np.eye(split_dim-ret_[-1].shape[0]))
    ret_ = np.stack(ret_)

    op = ms.ops.operations.CholeskyTrsm(split_dim=split_dim)
    ms0 = ms.Tensor(np0, dtype=ms.float32)
    ms1 = op(ms0).asnumpy()
    ret0 = np.einsum(ms1, [0,2,1], ms1, [0,2,3], [0,1,3])
    assert hfe(ret_, ret0) < 1e-4


class DummyNetDepend00(ms.nn.Cell):
    def __init__(self, tag_depend):
        super().__init__()
        self.global_step = ms.Parameter(ms.Tensor(0, dtype=ms.int32), name='global_step', requires_grad=False)
        self.tag_depend = tag_depend
    def construct(self, x):
        op_assign_step = ms.ops.assign(self.global_step, self.global_step+1)
        if self.tag_depend:
            x = ms.ops.depend(x, op_assign_step)
        ret = x * 233
        return ret

def test_depend00():
    np0 = np.random.randn(3).astype(np.float32)
    ms0 = ms.Tensor(np0, dtype=ms.float32)
    net0 = DummyNetDepend00(tag_depend=False)
    ret0 = net0(ms0)
    assert hfe(np0*233, ret0.asnumpy()) < 1e-5
    assert net0.global_step.asnumpy().item()==0

    net1 = DummyNetDepend00(tag_depend=True)
    ret1 = net1(ms0)
    assert hfe(np0*233, ret1.asnumpy()) < 1e-5
    assert net1.global_step.asnumpy().item()==1


class DummyNetDepend01(ms.nn.Cell):
    def __init__(self, N0, tag_depend):
        super().__init__()
        tmp0 = np.random.randn(N0).astype(np.float32)
        self.para0 = ms.Parameter(tmp0, name='para0', requires_grad=False)
        self.tag_depend = tag_depend
    def construct(self, x):
        op_assign_para0 = ms.ops.assign(self.para0, x)
        if self.tag_depend:
            tmp0 = ms.ops.depend(self.para0, op_assign_para0)
            ret = 0.233 * tmp0
        else:
            ret = 0.233 * self.para0
        return ret


def test_depend01(N0=3):
    np0 = np.random.randn(N0).astype(np.float32)
    ms0 = ms.Tensor(np0)
    ret_ = 0.233*np0
    net0 = DummyNetDepend01(N0, tag_depend=False)
    ret0 = net0(ms0).asnumpy()
    assert hfe(ret_, ret0) > 0.1 #mosttimes ret0 are just some random value

    net1 = DummyNetDepend01(N0, tag_depend=True)
    ret1 = net1(ms0).asnumpy()
    assert hfe(ret_, ret1) < 1e-5 #mosttimes ret0 are just some random value


class DummyNet02(ms.nn.Cell):
    def __init__(self, tag_pyassign):
        super().__init__()
        self.para0 = ms.Parameter(ms.Tensor(0, dtype=ms.float32), name='para0', requires_grad=False)
        self.tag_pyassign = tag_pyassign
        self.op = ms.ops.ReduceSum()
    def construct(self, x):
        if self.tag_pyassign:
            self.para0 = self.para0 + self.op(x) #strange behavior
        else:
            op_assign_para0 = ms.ops.assign(self.para0, self.para0+self.op(x))
            x = ms.ops.depend(x, op_assign_para0)
        ret = x * 233
        return ret

def test_parameter_python_assign():
    np0 = [np.random.randn(3).astype(np.float32) for _ in range(5)]
    ret_ = sum(x.sum() for x in np0)
    ms0 = [ms.Tensor(x,dtype=ms.float32) for x in np0]
    net0 = DummyNet02(tag_pyassign=False)
    for x in ms0:
        _ = net0(x)
    assert hfe(ret_, net0.para0.asnumpy()) < 1e-5
    assert isinstance(net0.para0, ms.Parameter)

    net1 = DummyNet02(tag_pyassign=True)
    for x in ms0:
        _ = net1(x)
    assert hfe(ret_, net1.para0.asnumpy()) < 1e-5
    assert isinstance(net1.para0, ms.Parameter)


class DummyNet03(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        import easydict
        self.ops = easydict.EasyDict()
        self.ops.mul = ms.ops.Mul()
        self.laji_mul = ms.ops.Mul()

    def construct(self, x, y):
        # ret = self.ops.mul(x, y) #fail
        # ret = getattr(self, 'laji_mul')(x,y) #fail
        ret = self.laji_mul(x, y)
        return ret

def demo_laji_ms_ops():
    ms0 = ms.Tensor(np.random.randn(3).astype(np.float32))
    ms1 = ms.Tensor(np.random.randn(3).astype(np.float32))
    net = DummyNet03()
    _ = net(ms0, ms1)


def demo_uniformreal_seed_behavior():
    # NEVER use seed and seed2 here
    op_uniform = ms.ops.UniformReal(seed=23, seed2=233)
    print(op_uniform((1,)).asnumpy()) #0.539
    print(op_uniform((1,)).asnumpy()) #0.539
    print(op_uniform((1,)).asnumpy()) #0.539

    op_uniform = ms.ops.UniformReal()
    print(op_uniform((1,)).asnumpy()) #0.266
    print(op_uniform((1,)).asnumpy()) #0.585
    print(op_uniform((1,)).asnumpy()) #0.268



@pytest.mark.skipif(ms.context.get_context('device_target')!='GPU', reason='GPU only')
def test_UpdateThorGradient(split_dim=3, N0=5, N1=7):
    np0 = np.random.randn(N0, split_dim, split_dim).astype(np.float32)
    np1 = np.random.randn(N0, split_dim, N1, split_dim).astype(np.float32)
    np2 = np.random.randn(N0, N1, split_dim, split_dim).astype(np.float32)
    ret_ = np.einsum(np0, [0,1,2], np1, [0,2,3,4], np2, [0,3,4,5], [0,1,3,5], optimize=True).reshape(N0*split_dim, N1*split_dim)

    ms0 = ms.Tensor(np0)
    ms1 = ms.Tensor(np1.reshape(N0*split_dim, -1).copy())
    ms2 = ms.Tensor(np2)
    op = ms.ops.operations.UpdateThorGradient(split_dim=split_dim)
    ret0 = op(ms0, ms1, ms2).asnumpy()
    assert hfe(ret_, ret0) < 1e-4


@pytest.mark.skipif(ms.context.get_context('device_target')!='Ascend', reason='Ascend only')
def test_CusMatMulCube(N0=3, N1=5, N2=7):
    np0 = np.random.randn(N0, N1).astype(np.float32)
    np1 = np.random.randn(N1, N2).astype(np.float32)
    ret_ = np0 @ np1
    ms0 = ms.Tensor(np0)
    ms1 = ms.Tensor(np1)
    op0 = ms.ops.operations.CusMatMulCube()
    op1 = ms.ops.MatMul()
    ret0 = op0(ms0, ms1).asnumpy()
    ret1 = op1(ms0, ms1).asnumpy()
    assert hfe(ret_, ret0) < 0.005 #maybe float16 is used in calculation
    assert hfe(ret_, ret1) < 1e-5


@pytest.mark.skipif(ms.context.get_context('device_target')!='Ascend', reason='Ascend only')
def test_CusBatchMatMul():
    N0 = random.choice([1,2,4,8,16]) #fail for N0=3,5,6
    N1 = 128 #128 only
    np0 = np.random.randn(N0, N1, N1).astype(np.float16)
    np1 = np.random.randn(N0, N1, N1).astype(np.float16)
    ret_ = np.einsum(np0, [0,1,2], np1, [0,3,2], [0,1,3], optimize=True)

    ms0 = ms.Tensor(np0)
    ms1 = ms.Tensor(np1)
    op0 = ms.ops.operations.CusBatchMatMul()
    ret0 = op0(ms0, ms1).asnumpy()
    assert hfe(ret_, ret0) < 0.01 #strange


@pytest.mark.skipif(ms.context.get_context('device_target')!='Ascend', reason='Ascend only')
def test_CusCholeskyTrsm():
    hf_definite = lambda x: np.matmul(x, x.T)
    N0 = random.choice([1,2,4,8,16]) #fail for N0=3,5,6, large error for N0=1,2
    split_dim = 128
    np0 = hf_definite(np.random.randn(N0*split_dim, N0*split_dim).astype(np.float32)) + np.eye(split_dim)*20

    tmp0 = np0.reshape(N0,split_dim,N0,split_dim)
    ret0_ = np.stack([np.linalg.inv(np.linalg.cholesky(tmp0[x,:,x]).T) for x in range(N0)])
    ret1_ = np.matmul(ret0_, ret0_.transpose(0,2,1))
    # ret1_ = np.einsum(ret0_, [0,1,2], ret0_, [0,3,2], [0,1,3])

    op0 = ms.ops.operations.CusCholeskyTrsm() #split_dim must be 128
    op1 = ms.ops.operations.CusBatchMatMul()
    # op1 = ms.ops.operations.BatchMatMul(transpose_b=True) #still dtype error
    ms0 = ms.Tensor(np0, dtype=ms.float32)
    tmp0 = op0(ms0)
    ret0 = tmp0.asnumpy()
    ret1 = op1(tmp0, tmp0).asnumpy()
    print(hfe(ret0_, ret0), hfe(ret1_, ret1))
    assert hfe(ret0_, ret0) < 1e-4 #sometimes small sometimes large, ┑(￣Д ￣)┍
    assert hfe(ret1_, ret1) < 1e-4 #sometimes small sometimes large, ┑(￣Д ￣)┍
