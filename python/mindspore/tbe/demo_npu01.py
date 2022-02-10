import numpy as np
import mindspore as ms
import te
import topi

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="Ascend")

# https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/advanced_use/custom_operator_ascend.html

@te.platform.fusion_manager.fusion_manager.register("add3")
def add3_compute(input1, input2, const_bias):
    sum2 = te.lang.cce.vadd(input1, input2)
    sum3 = te.lang.cce.vadds(sum2, te.tvm.const(const_bias, dtype=input1.dtype))
    return sum3


cus_add3_op_info = ms.ops.TBERegOp("CusAdd3") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("add3.so") \
    .compute_cost(10) \
    .kernel_name("CusAdd3Impl") \
    .partial_flag(True) \
    .attr("const_bias", "required", "float", "all") \
    .input(0, "input1", False, "required", "all") \
    .input(1, "input2", False, "required", "all") \
    .output(0, "sum", False, "required", "all") \
    .dtype_format(ms.ops.DataType.F32_Default, ms.ops.DataType.F32_Default, ms.ops.DataType.F32_Default) \
    .dtype_format(ms.ops.DataType.F16_Default, ms.ops.DataType.F16_Default, ms.ops.DataType.F16_Default) \
    .get_op_info()


@ms.ops.op_info_register(cus_add3_op_info)
def CusAdd3Impl(input1, inptu2, sum1, const_bias, kernel_name="CusAdd3Impl"):
    shape = input1.get("shape")
    shape = topi.cce.util.shape_refine(shape)
    dtype = input1.get("dtype").lower()
    input1 = te.tvm.placeholder(shape, name="input1", dtype=dtype.lower())
    input2 = te.tvm.placeholder(shape, name="input2", dtype=dtype.lower())

    with te.tvm.target.cce():
        res = add3_compute(input1, input2, const_bias)
        sch = topi.generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [input1, input2, res]}

    te.lang.cce.cce_build_code(sch, config)

# sum = input1 + input2 + const_bias
class CusAdd3(ms.ops.PrimitiveWithInfer):

    @ms.ops.prim_attr_register
    def __init__(self, const_bias=0.0):
        self.CusAdd3Impl = CusAdd3Impl
        self.init_prim_io_names(inputs=['input1', 'input2'], outputs=['sum3'])

    def infer_shape(self, input1, input2):
        return input1

    def infer_dtype(self, input1, input2):
        return input1


if __name__=='__main__':
    np0 = np.random.randn(3).astype(np.float32)
    np1 = np.random.randn(3).astype(np.float32)
    bias = 0.233
    ret_ = np0 + np1 + bias
    op = CusAdd3(bias)
    ret0 = op(ms.Tensor(np0), ms.Tensor(np1)).asnumpy()
    assert hfe(ret_, ret0) < 1e-5
