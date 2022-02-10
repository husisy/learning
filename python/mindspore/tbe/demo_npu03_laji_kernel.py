import numpy as np
import mindspore as ms
import te
import topi
import te.tik

cus_matrix_combine_op_info = ms.ops.TBERegOp("CusMatrixCombine") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("matrixcombine.so") \
    .compute_cost(10) \
    .kernel_name("CusMatrixCombine") \
    .partial_flag(True) \
    .input(0, "x1", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(ms.ops.DataType.F32_Default, ms.ops.DataType.F32_Default) \
    .get_op_info()


@ms.ops.op_info_register(cus_matrix_combine_op_info)
def CusMatrixCombine(input_x, output, kernel_name="matrix_combine"):
    input_x_shape = input_x.get("shape")
    output_shape = output.get("shape")
    split_dim = 128

    if topi.cce.util.get_product_version() == topi.cce.util.VERSION_MINI:
        tik_instance = te.tik.Tik(te.tik.Dprofile("v100", "mini"))
    else:
        tik_instance = te.tik.Tik(te.tik.Dprofile("v100", "cloud"))

    input_x = tik_instance.Tensor("float32", input_x_shape, name="input_x", scope=te.tik.scope_gm)
    res = tik_instance.Tensor("float32", output_shape, name="res", scope=te.tik.scope_gm)

    blocks = 32
    matrix_dim = input_x_shape[0] * input_x_shape[1]
    if input_x_shape[0] == 1 and input_x_shape[1] == 64:
        tiling_dim = 2
        bs = 1
        with tik_instance.for_range(0, blocks, block_num=blocks) as block_index:
            input_x_ub = tik_instance.Tensor("float32", (tiling_dim, matrix_dim), name="input_x_ub", scope=te.tik.scope_ubuf)
            tik_instance.data_move(input_x_ub, input_x[0, block_index * tiling_dim, 0], 0, 1, 16, 0, 0)
            tik_instance.data_move(res[block_index * tiling_dim, 0], input_x_ub, 0, 1, 16, 0, 0)
    else:
        tiling_dim = 4
        bs = input_x_shape[0]
        with tik_instance.for_range(0, blocks, block_num=blocks) as block_index:
            input_x_ub = tik_instance.Tensor("float32", (tiling_dim, matrix_dim), name="input_x_ub", scope=te.tik.scope_ubuf)
            zero = tik_instance.Scalar("float32")
            zero.set_as(0.0)
            with tik_instance.for_range(0, bs) as i:
                repeat_real = tiling_dim * matrix_dim // 64
                if repeat_real <= 255:
                    tik_instance.vector_dup(64, input_x_ub, zero, repeat_real, 1, 8)
                else:
                    repeat_1 = 255
                    repeat_2 = repeat_real - 255
                    tik_instance.vector_dup(64, input_x_ub, zero, repeat_1, 1, 8)
                    tik_instance.vector_dup(64, input_x_ub[255 * 64], zero, repeat_2, 1, 8)
                with tik_instance.for_range(0, tiling_dim) as j:
                    tik_instance.data_move(input_x_ub[j,split_dim*i], input_x[i,block_index*tiling_dim+j, 0], 0, 1, 16, 0, 0)
                tmp0 = res[i*split_dim+block_index*tiling_dim, 0]
                tik_instance.data_move(tmp0, input_x_ub, 0, 1, tiling_dim*matrix_dim*4//32, 0, 0)
    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[input_x], outputs=[res])
    return tik_instance
