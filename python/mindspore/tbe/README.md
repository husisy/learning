# TBE

1. link
   * [TBE算子开发框架](https://support.huaweicloud.com/odevg-Inference-cann/atlaste_10_0013.html)
   * `mindspore/ops/_op_impl_custom_op/`, `mindspore/ops/operations/_thor_ops.py`
   * [mindspore-自定义算子-ascend](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/advanced_use/custom_operator_ascend.html)
2. Tensor Virtual Machine (TVM)：开源深度学习编译栈，它通过统一的中间表达（Intermediate Representation）堆栈连接深度学习模型和后端硬件平台，通过统一的结构优化Schedule，可以支持CPU、GPU和特定的加速器平台和语言, [official-site](https://tvm.apache.org/)
3. Tensor Boost Engine (TBE)
4. 限制
   * 当前仅支持用户开发Vector算子，由于开发高性能Cube算子难度较大，暂由华为交付
   * 当前暂不开放Unified Buffer与L1 Buffer之间的通路
5. 调度过程主要包括数据流管理、tiling以及指令映射等
6. Tensor Iterator Kernel (TIK)
