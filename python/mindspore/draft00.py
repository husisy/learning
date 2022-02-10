import numpy as np
import mindspore as ms

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target='GPU')
# CPU is not supported in Pynative mode, ms.common.api.ms_function


hf0 = lambda x,y: x*y
ms0 = ms.Tensor(1, ms.float32)
ms1 = ms.Tensor(2, ms.float32)
ms0_grad,ms1_grad = ms.ops.GradOperation(get_all=True)(hf0)(ms0, ms1)


# one-to-one mapping from np.dtype
ms.int8 #int16 int32(default) int64
ms.float16 #float32(default) float64
ms.uint8 #uint16 uint32 uint64
ms.bool_
ms.dtype_to_nptype(ms.float32)
ms.dtype_to_pytype(ms.int8) #int
ms.pytype_to_dtype(int) #int64

ms0 = ms.Tensor(np.random.rand(2,3)) #ms.float64
ms.ops.dtype(ms0) #ms.float64
ms1 = ms.ops.cast(ms0, ms.float32) #out-of place
ms0.set_dtype(ms.float32) #in-place


# create tensor
ms.Tensor(np.array([[1, 2], [3, 4]]), ms.int32)
ms.Tensor(1.0, ms.int32)
ms.Tensor(2, ms.int32)
ms.Tensor(True, ms.bool_)
ms.Tensor((1, 2, 3), ms.int16)
ms.Tensor([4.0, 5.0, 6.0], ms.float64)

ms0 = ms.Tensor([4.0, 5.0, 6.0], ms.float64)
ms0.shape
ms0.dtype
ms0.dim()
ms0.size()
ms0.asnumpy()
ms0.set_dtype(ms.float32) #modify in-place
# ms0.any() #Ascend only
# ms0.all() #Ascend only


# ms.ops
ms0 = ms.Tensor(np.random.rand(3,4).astype(np.float32))
ms1 = ms.ops.reshape(ms0, (4,3)) #float64 fail on mindspore-1.0.1
# ms.ops.Reshape()

ms.nn.Softmax
ms.nn.ReLU
ms.nn.ELU
ms.nn.Tanh
ms.nn.Sigmoid
ms.nn.Dense
ms.nn.Flatten
ms.nn.Dropout
ms.nn.Norm
ms.nn.OneHot
ms.nn.SequentialCell
ms.nn.CellList
ms.nn.Conv2d
ms.nn.Conv1d
ms.nn.Conv2dTranspose
ms.nn.Conv1dTranspose
ms.nn.AvgPool2d

ms.ops.shape(ms.Tensor(np.random.rand(2,3)))
# ms.ops.Shape

ms0 = ms.Tensor(np.ones([2, 8]).astype(np.float32))
ms1 = ms.ops.Broadcast(1)((ms0,)) #Ascend only

ms0 = ms.Tensor(np.random.rand(3,4,5).astype(np.float32))
ms1 = ms.ops.Transpose()(ms0, (2,1,0))

np0 = np.random.rand(3,7).astype(np.float32)
np1 = np.random.rand(3,7).astype(np.float32)
ms0 = ms.ops.Concat(axis=0)((ms.Tensor(np0), ms.Tensor(np1))) #(6,7)
ms1 = ms.ops.Concat(axis=1)((ms.Tensor(np0), ms.Tensor(np1))) #(3,14)

ms.ops.TopK()
ms.ops.Argmax()
ms.ops.Square()
ms.ops.Slice()
ms.ops.Cast() #ms.ops.cast()
ms.ops.Abs()
ms.ops.Pow() #ms.ops.tensor_pow()
ms.ops.ReLU() #ms.nn.ReLU()
ms.ops.SoftmaxCrossEntropyWithLogits() #ms.nn.SoftmaxCrossEntropyWithLogits()
ms.ops.ApplyMomentum() #ms.nn.Momentum()

ms0 = ms.ops.normal((2,3), mean=ms.Tensor(0,ms.float32), stddev=ms.Tensor(1,ms.float32))


multi_type_add = ms.ops.MultitypeFuncGraph('add')
@multi_type_add.register('Number', 'Number')
def add_scalar(x, y):
    return ms.ops.scalar_add(x, y)
@multi_type_add.register('Tensor', 'Tensor')
def add_tensor(x, y):
    return ms.ops.tensor_add(x, y)

ms0 = ms.Tensor(np.random.randn(2,2).astype(np.float32))
ms1 = ms.Tensor(np.random.randn(2,2).astype(np.float32))
multi_type_add(ms0, ms1)
multi_type_add(1, 2)


ms.ops.ACos()(ms.Tensor(np.random.rand(2,3).astype(np.float32))) #Ascend only


# ms.nn.SGD


batch_size = 3
num_box = 5
image = ms.Tensor(np.random.normal(size=[batch_size, 256, 256, 3]).astype(np.float32))
box = ms.Tensor(np.random.rand(num_box, 4).astype(np.float32))
box_index = ms.Tensor(np.random.randint(0, batch_size, size=(num_box), dtype=np.int32))
crop_and_resize = ms.ops.CropAndResize()
z0 = crop_and_resize(image, box, box_index, (24,24)) #(5,24,24,3)


# TODO add test
class DebugNN(ms.nn.Cell):
    def __init__(self,):
        self.debug = nn.Debug()

    def construct(self, x, y):
        self.debug()
        x = self.add(x, y)
        self.debug(x)
        return x


# PyNative-only
def hook_fn(grad_out):
    print(grad_out)
grad_all = ms.ops.GradOperation(get_all=True)
def hook_test(x, y):
    x = ms.ops.HookBackward(hook_fn)(x)
    z = x * y
    return z
grad_all(hook_test)(ms.Tensor(1, ms.float32), ms.Tensor(2, ms.float32))


## parameter, MetaTensor
ms0 = ms.Parameter(ms.Tensor(np.random.rand(2,3)), name='ms0') #(ms,float64,(2,3))
tmp0 = ms.common.initializer.initializer('ones', [1, 2, 3], ms.float32)
ms1 = ms.Parameter(tmp0, name='ms1') #(ms,float32,(1,2,3))
ms2 = ms.Parameter(2.0, name='ms2') #(ms,float32,(,))
ms0.name
ms0.is_init
ms0.requires_grad
ms0.data #id(ms0)==id(ms0.data) #Err....
ms.Tensor(ms0) #convert to ms.Tensor
# ms0.init_data()
# ms0.set_data()
# ms0.set_param_ps()
ms3 = ms0.clone(prefix='clone') #clone.ms0
