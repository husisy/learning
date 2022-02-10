import numpy as np
import cupy as cp

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

assert cp.cuda.is_available()


def test_cp_asarray():
    cp0 = cp.random.randn(1000)
    cp1 = cp.asarray(cp0) #cp.asarray() does NOT copy the input if possible
    cp2 = cp.array(cp0) #cp.array() copy the input by default, unless cp.array(x, copy=False)
    assert cp.shares_memory(cp0, cp1, max_work='MAY_SHARE_BOUNDS')==True
    assert cp.shares_memory(cp0, cp2, max_work='MAY_SHARE_BOUNDS')==False


def test_np_asarray():
    np0 = np.random.randn(1000)
    np1 = np.asarray(np0)
    np2 = np.array(np0)
    assert np.shares_memory(np0, np1)==True
    assert np.shares_memory(np0, np2)==False


def test_np_cp_conversion():
    np0 = np.random.randn(1000)
    np1 = cp.asnumpy(cp.asarray(np0))
    np2 = cp.array(np0).get()
    assert hfe(np0, np1) < 1e-7
    assert hfe(np0, np2) < 1e-7


def test_cp_linalg_norm():
    np0 = np.random.randn(1000)
    cp0 = cp.array(np0)
    ret_ = np.linalg.norm(np0)
    ret0 = cp.linalg.norm(cp0)
    assert hfe(ret_, cp.asnumpy(ret0)) < 1e-7


def xp_softplus(x):
    xp = cp.get_array_module(x)
    return xp.maximum(0,x) + xp.log1p(xp.exp(-abs(x)))

def test_np_cp_agnostic_code():
    np0 = np.random.randn(1000)
    np1 = xp_softplus(np0)
    np2 = cp.asnumpy(xp_softplus(cp.asarray(np0)))
    assert hfe(np1, np2) < 1e-7


def test_ElementwiseKernel():
    cp0 = cp.random.rand(300)
    cp1 = cp.random.rand(300) #support broadcast
    ret_ = (cp0-cp1)**2
    # input, output, kernel, name
    # variable name "n" "i" "_" is reserved
    hf0 = cp.ElementwiseKernel('float64 x, float64 y', 'float64 z', 'z = (x - y) * (x - y)', 'squared_diff')
    ret0 = hf0(cp0, cp1)
    ret1 = cp.empty((300,), dtype=np.float64)
    hf0(cp0, cp1, ret1)
    assert hfe(cp.asnumpy(ret_), cp.asnumpy(ret0)) < 1e-7
    assert hfe(cp.asnumpy(ret_), cp.asnumpy(ret1)) < 1e-7


hf_polyval_ekernel = cp.ElementwiseKernel(
    in_params='T x, raw T y',
    out_params='T z',
    operation='z=x*y[0] + x*x*y[1] + x*x*x*y[2];',
    name='polyval_ekernel',
)
def test_cp_ElmeentwiseKernel_raw(N0=500):
    np0 = np.random.randn(N0) + np.random.randn(N0)*1j
    np1 = np.random.randn(3) + np.random.randn(3)*1j
    ret_ = np0*np1[0] + np0**2*np1[1] + np0**3*np1[2]
    ret0 = hf_polyval_ekernel(cp.array(np0), cp.array(np1))
    assert hfe(ret_, ret0.get()) < 1e-7


def test_complex_real_imag_part(N0=233):
    np0 = np.random.randn(N0) + 1j*np.random.randn(N0)
    ret_ = np0**2

    hf0 = cp.ElementwiseKernel('T x0, T x1', 'T y0, T y1', 'T tmp0=x0*x0-x1*x1; y1=2*x0*x1; y0=tmp0;', 'hf0_ekernel')
    cp0 = cp.array(np0)
    hf0(cp0.real, cp0.imag, cp0.real, cp0.imag)
    assert hfe(ret_, cp0.get()) < 1e-7


def test_reduction_kerenl(N0=3, N1=5):
    np0 = np.random.rand(N0,N1)
    np1 = np.random.rand(N0,N1)
    ret_ = np.sqrt(np.sum(np0*np1, axis=1))

    hf0 = cp.ReductionKernel(
        in_params='T x0, T x1',
        out_params='T y',
        map_expr='x0 * x1',
        reduce_expr='a + b',
        post_map_expr='y = sqrt(a)',
        identity='0',
        name='hf0_rkernel',
    )
    cp0 = cp.array(np0)
    cp1 = cp.array(np1)
    ret0 = hf0(cp0, cp1, axis=1)
    assert hfe(ret_, ret0.get()) < 1e-7


def test_cupy_for_loop_ekernel():
    tmp0 = 'T tmp0=0; for(int i=0; i<N0; i++){ tmp0 += x0*x1[i]; } r0 = tmp0;'
    hf0_ekernel = cp.ElementwiseKernel(in_params='T x0, raw T x1, int32 N0', out_params='T r0', operation=tmp0, name='hf0_ekernel')
    np0 = np.random.rand(3,5)
    np1 = np.random.rand(7)
    ret_ = np0*np1.sum()
    cp0 = cp.array(np0)
    cp1 = cp.array(np1)
    ret0 = hf0_ekernel(cp0, cp1, cp1.shape[0])
    assert hfe(ret_, ret0.get())
