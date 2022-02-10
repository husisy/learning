import te
import topi
import te.tik


# https://support.huaweicloud.com/odevg-Inference-cann/atlaste_10_0017.html
shape_x = (2,3)
shape_y = (2,3)
input_data_type = 'float32'
kernel_name = 'test_kernel233'
data_x = te.tvm.placeholder(shape_x, name="data_1", dtype=input_data_type)
data_y = te.tvm.placeholder(shape_y, name="data_2", dtype=input_data_type)
res = te.lang.cce.vadd(data_x, data_y)
with te.tvm.target.cce():
    schedule = topi.generic.auto_schedule(res)
config = {"print_ir":False, "name":kernel_name, "tensor_list": (data_x, data_y, res)}
te.lang.cce.cce_build_code(schedule, config)


# https://support.huaweicloud.com/odevg-Inference-cann/atlaste_10_0019.html
x = te.tvm.placeholder((512, 1024), 'float16')
exp_x = te.lang.cce.vexp(x)
reduce_exp_x = te.lang.cce.sum(exp_x, axis=0)
res = te.lang.cce.vrec(reduce_exp_x)
with te.tvm.target.cce():
    sch = topi.generic.auto_schedule(res)

tik_instance = te.tik.Tik()
data_A = tik_instance.Tensor("float16", (128,), name="data_A", scope=te.tik.scope_gm)
data_B = tik_instance.Tensor("float16", (128,), name="data_B", scope=te.tik.scope_gm)
data_C = tik_instance.Tensor("float16", (128,), name="data_C", scope=te.tik.scope_gm)
data_A_ub = tik_instance.Tensor("float16", (128,), name="data_A_ub", scope=te.tik.scope_ubuf)
data_B_ub = tik_instance.Tensor("float16", (128,), name="data_B_ub", scope=te.tik.scope_ubuf)
data_C_ub = tik_instance.Tensor("float16", (128,), name="data_C_ub", scope=te.tik.scope_ubuf)
tik_instance.data_move(data_A_ub, data_A, 0, 1, 128 //16, 0, 0)
tik_instance.data_move(data_B_ub, data_B, 0, 1, 128 //16, 0, 0)
repeat = tik_instance.Scalar('int32')
repeat.set_as(1)
tik_instance.vec_add(128, data_C_ub[0], data_A_ub[0], data_B_ub[0], repeat, 8, 8, 8)
tik_instance.data_move(data_C, data_C_ub, 0, 1, 128 //16, 0, 0)
tik_instance.BuildCCE(kernel_name="test_kernel234",inputs=[data_A,data_B],outputs=[data_C])
