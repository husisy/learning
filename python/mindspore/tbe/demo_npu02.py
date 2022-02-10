import numpy as np
import mindspore as ms
import te
import topi
import te.tik

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="Ascend")

# https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/advanced_use/custom_operator_ascend.html


cus_add2_op_info = ms.ops.TBERegOp("CusAdd2") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("add2.so") \
    .compute_cost(10) \
    .kernel_name("CusAdd2Impl") \
    .partial_flag(True) \
    .input(0, "input1", False, "required", "all") \
    .input(1, "input2", False, "required", "all") \
    .output(0, "data_C", False, "required", "all") \
    .dtype_format(ms.ops.DataType.F32_Default, ms.ops.DataType.F32_Default, ms.ops.DataType.F32_Default) \
    .dtype_format(ms.ops.DataType.F16_Default, ms.ops.DataType.F16_Default, ms.ops.DataType.F16_Default) \
    .get_op_info()


@ms.ops.op_info_register(cus_add2_op_info)
def CusAdd2Impl(input1, inptu2, sum1, kernel_name="CusAdd2Impl"):
    tik_instance = te.tik.Tik()
    input1 = tik_instance.Tensor("float16", (128,), name="input1", scope=te.tik.scope_gm)
    input2 = tik_instance.Tensor("float16", (128,), name="input2", scope=te.tik.scope_gm)
    data_C = tik_instance.Tensor("float16", (128,), name="data_C", scope=te.tik.scope_gm)
    data_A_ub = tik_instance.Tensor("float16", (128,), name="data_A_ub", scope=te.tik.scope_ubuf)
    data_B_ub = tik_instance.Tensor("float16", (128,), name="data_B_ub", scope=te.tik.scope_ubuf)
    data_C_ub = tik_instance.Tensor("float16", (128,), name="data_C_ub", scope=te.tik.scope_ubuf)
    tik_instance.data_move(data_A_ub, input1, 0, 1, 128 //16, 0, 0)
    tik_instance.data_move(data_B_ub, input2, 0, 1, 128 //16, 0, 0)
    repeat = tik_instance.Scalar('int32')
    repeat.set_as(1)
    tik_instance.vec_add(128, data_C_ub[0], data_A_ub[0], data_B_ub[0], repeat, 8, 8, 8)
    tik_instance.data_move(data_C, data_C_ub, 0, 1, 128 //16, 0, 0)
    tik_instance.BuildCCE(kernel_name=kernel_name,inputs=[input1,input2],outputs=[data_C])

class CusAdd2(ms.ops.PrimitiveWithInfer):

    @ms.ops.prim_attr_register
    def __init__(self):
        self.CusAdd2Impl = CusAdd2Impl
        self.init_prim_io_names(inputs=['input1', 'input2'], outputs=['data_C'])

    def infer_shape(self, input1, input2):
        return input1

    def infer_dtype(self, input1, input2):
        return input1


if __name__=='__main__':
    np0 = np.random.randn(128).astype(np.float16)
    np1 = np.random.randn(128).astype(np.float16)
    ret_ = np0 + np1
    op = CusAdd2()
    ret0 = op(ms.Tensor(np0), ms.Tensor(np1)).asnumpy()
    assert hfe(ret_, ret0) < 1e-5
