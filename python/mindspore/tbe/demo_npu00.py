import numpy as np
import mindspore as ms
import te
import topi

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="Ascend")

# https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/advanced_use/custom_operator_ascend.html

@te.platform.fusion_manager.fusion_manager.register("square")
def square_compute(input_x, output_y):
    """
    calculating data's square,y= x*x

    input_x(tvm.tensor): the placeholder of input data
    output_y(dict): shape and dtype of output, should be same shape and type as input
    (ret)(tvm.tensor): the result of square
    """
    ret = te.lang.cce.vmul(input_x, input_x)
    return ret


cus_square_op_info = ms.ops.TBERegOp("CusSquare") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("square.so") \
    .compute_cost(10) \
    .kernel_name("CusSquareImpl") \
    .partial_flag(True) \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(ms.ops.DataType.F32_Default, ms.ops.DataType.F32_Default) \
    .dtype_format(ms.ops.DataType.F16_Default, ms.ops.DataType.F16_Default) \
    .get_op_info()


@ms.ops.op_info_register(cus_square_op_info)
def CusSquareImpl(input_x, output_y, kernel_name="CusSquareImpl"):
    shape = input_x.get("shape")
    dtype = input_x.get("dtype").lower()
    shape = topi.cce.util.shape_refine(shape)
    data = te.tvm.placeholder(shape, name="data", dtype=dtype.lower())
    with te.tvm.target.cce():
        res = square_compute(data, output_y)
        sch = topi.generic.auto_schedule(res)
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, res]}
    te.lang.cce.cce_build_code(sch, config)


class CusSquare(ms.ops.PrimitiveWithInfer):

    @ms.ops.prim_attr_register
    def __init__(self):
        self.CusSquareImpl = CusSquareImpl #NECESSARY
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def infer_shape(self, data_shape):
        return data_shape

    def infer_dtype(self, data_dtype):
        return data_dtype

    def get_bprop(self):
        def bprop(data, out, dout):
            dx = 2 * data * dout
            return (dx,)
        return bprop


class Net(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.square = CusSquare()

    def construct(self, data):
        return self.square(data)

if __name__=='__main__':
    np0 = np.random.randn(3).astype(np.float32)
    ms0 = ms.Tensor(np0, dtype=ms.float32)
    op = CusSquare()
    ms1 = op(ms0)
    assert hfe(np0**2, ms1.asnumpy()) < 1e-5

    np0 = np.random.randn(3).astype(np.float32)
    np1 = np.random.randn(3).astype(np.float32)
    ms0 = ms.Tensor(np0, dtype=ms.float32)
    ms1 = ms.Tensor(np1, dtype=ms.float32)
    net = Net()
    # op = CusSquare()
    grad_with_sens = ms.ops.GradOperation(sens_param=True)
    ms2 = grad_with_sens(net)(ms0, ms1)
    assert hfe(2*np0*np1, ms2.asnumpy()) < 1e-5
