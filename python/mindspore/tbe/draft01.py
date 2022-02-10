import te
import numpy as np


# https://support.huaweicloud.com/odevg-Inference-cann/atlaste_10_0064.html

# 比较两个输入的每个维度的大小，取每个维度的大值，生成out_shape
def _produce_shapes(shape1, shape2):
    shape1 = list(shape1)
    shape2 = list(shape2)
    flag = 0
    if len(shape1) < len(shape2):
        shape1, shape2 = shape2, shape1
        flag = 1

    output_shape_len = len(shape1)
    dec = output_shape_len - len(shape2)
    for i in range(dec):
        shape2 = [1] + shape2

    out_shape = []
    for i in range(output_shape_len):
        if (shape1[i] != shape2[i]) and (shape1[i] != 1) and (shape2[i] != 1):
            raise RuntimeError("input shapes not match!")
        out_shape.append(shape1[i] if shape1[i] > shape2[i] else shape2[i])

    if flag == 1:
        shape1, shape2 = shape2, shape1

    return shape1, shape2, out_shape

# 将shape转换为list
def _shape_to_list(shape):
    result = []
    for i in shape:
        if isinstance(i, tvm.expr.Var):
            result.append(i)
        else:
            result.append(i.value)
    return result

# 实现Add算子的计算逻辑
@te.platform.fusion_manager.register("add")
def add_compute(input_x, input_y, output_z, kernel_name="add"):
    shape_x = _shape_to_list(input_x.shape)
    shape_y = _shape_to_list(input_y.shape)
    shape_x, shape_y, shape_max = _produce_shapes(shape_x, shape_y)   # shape_max取shape_x与shape_y的每个维度的大值
    shape_size = reduce(lambda x, y: x * y, shape_max[:])
    assert shape_size <= 2**31

    input_x = te.lang.cce.broadcast(input_x, shape_max)       # 将input_x的shape广播为shape_max
    input_y = te.lang.cce.broadcast(input_y, shape_max)       # 将input_y的shape广播为shape_max
    ret = te.lang.cce.vadd(input_x, input_y)        # 执行input_x + input_y

    return ret

# 算子定义函数
def add(input_x, input_y, output_z, kernel_name="add"):
    # 获取算子输入tensor的shape与dtype
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")
    shape_x = (2,3)
    shape_y = (3,)
    dtype = 'float32'
    assert dtype in {'float16','float32','int32'}
    # shape_max取shape_x与shape_y的每个维度的最大值
    shape_x, shape_y, shape_max = _produce_shapes(shape_x, shape_y)
    if shape_x[-1] == 1 and shape_y[-1] == 1 and shape_max[-1] == 1:
        # 如果shape的长度等于1，就直接赋值，如果shape的长度不等于1，做切片，将最后一个维度舍弃（按照内存排布，最后一个维度为1与没有最后一个维度的数据排布相同，例如2*3=2*3*1，将最后一个为1的维度舍弃，可提升后续的调度效率）。
        shape_x = shape_x if len(shape_x) == 1 else shape_x[:-1]
        shape_y = shape_y if len(shape_y) == 1 else shape_y[:-1]
        shape_max = shape_max if len(shape_max) == 1 else shape_max[:-1]


    # 使用TVM的placeholder接口对第一个输入tensor进行占位，返回一个tensor对象
    data_x = tvm.placeholder(shape_x, name="data_1", dtype=dtype)
    # 使用TVM的placeholder接口对第二个输入tensor进行占位，返回一个tensor对象
    data_y = tvm.placeholder(shape_y, name="data_2", dtype=dtype)

    # 调用compute实现函数
    res = add_compute(data_x, data_y, output_z, kernel_name)
    # 自动调度
    with tvm.target.cce():
        schedule = generic.auto_schedule(res)
    # 编译配置
    config = {"name": kernel_name,
              "tensor_list": (data_x, data_y, res)}
    te.lang.cce.cce_build_code(schedule, config)
