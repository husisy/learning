# mindspore

1. link
   * [official-site](https://www.mindspore.cn/)
   * [documentation](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
   * [gitee](https://gitee.com/mindspore/mindspore)
   * [mindspore算子支持](https://www.mindspore.cn/doc/note/zh-CN/r1.0/operator_list_ms.html)
   * [mindspore入门](https://gitee.com/mindspore/course/blob/master/mindspore/README.md)
   * [优化算法](https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.0/optim.html)，未出现在常规文档链接上
2. installation见单独仓库
3. 偏见
   * `ms.dataset.vision.c_transforms`和`ms.dataset.vision.py_trainsforms`**禁止**混用
4. 总体架构
   * 前端表示层MindExpression (ME)：用户级应用软件编程接口，用于科学计算以及构建和训练神经网络，并将用户的Python代码转换为数据流图
   * 计算图引擎GraphEngine (GE)：算子和硬件资源的管理器，负责控制从ME接收的数据流图的执行
   * 后端运行时：云、边、端上不同环境中的高效运行环境，例如CPU、GPU、Ascend AI处理器、 Android/iOS等
5. 源码转换Source Code Transformation (SCT)
6. MindSight
7. API层次结构
   * low-level: `Tensor,Parameter,Ops`
   * medium-level: `Cell,Loss,Optimizer,Layers,Dataset`
   * high-level: `Model,Profiler,Callback,Amp,Quant`
8. 模块：mindspore extend, mindExpress, mindCompiler
9. 算子原语
10. context：执行模式管理，硬件管理，分布式管理，维测profile管理
    * 执行模式管理`mode`：PyNative, Graph。`ms.context.PYNATIVE_MODE`, `ms.context.GRAPH_MODE`，可随时切换
    * 硬件管理`device_target/device_id`：`device_id`对于CPU/GPU无效
    * 分布式管理见下
    * 维测管理profile见下
11. 模型保存数据格式：`MINDIR/CheckPoint/AIR/ONNX`
    * `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` protobuf文件大小限制`64M`
    * 分布式训练时，每个进程**务必**设置不同的文件夹保存参数，避免读写错乱
    * `CheckPointConfig`默认保存最后一个step的CheckPoint文件
12. callback: `ModelCheckpoint/SummaryCollector/LossMonitor/TimeMonitor`
13. TODO
    * [doc/优化数据处理](https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/advanced_use/optimize_data_processing.html)

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/zhangc/miniconda3/envs/mindspore/lib"
rm /usr/local/cuda/version.txt
```

## distributed

1. link
   * [documentation/分布式训练](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/distributed_training_ascend.html)
   * [documentation/分布式训练设计](https://www.mindspore.cn/doc/note/zh-CN/master/design/mindspore/distributed_training_design.html)
   * [使用Parameter Server训练](https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/advanced_use/apply_parameter_server_training.html)
2. 单机场景下支持`1/2/4/8`卡设备集群，多机场景下支持`8*n`卡设备集群
3. 每台机器的`0-3`卡和`4-7`卡各为1个组网，2卡和4卡训练时卡必须相连且不支持跨组网创建集群
4. 分布式管理`ms.context.set_auto_parallel_context`：必须在初始化网络之前调用
   * `parallel_mode`: `STAND_ALONE/DATA_PARALLEL/AUTO_PARALLEL`
   * `gradients_mean`
   * `enable_parallel_optimizer`：只在数据并行模式和参数量大于机器数时有效
   * `device_num/global_rank`建议保留默认值，mindspore框架会调用HCCL接口来获取
5. TODO
   * 对比多进程保存的ckpt文件是否一致
   * 测试多进程`all-reduce`
   * 测试BN
   * [doc/自动混合精度](https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/advanced_use/enable_mixed_precision.html)
   * [doc/LSTM](https://gitee.com/mindspore/mindspore/tree/r1.0/model_zoo/official/nlp/lstm)，静态图变长度tensor
   * [doc/initializer](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.common.initializer.html)

## profile

1. link
   * [doc/执行管理/维测管理](https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.0/context.html#id8)
   * [doc/自定义调试信息](https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/advanced_use/custom_debugging_info.html)

## mindinsight

1. `pip install`的详细安装链接可以查看「mindspore官网-安装-查看版本列表和接口变更」
2. `mindinsight`运行在后台，使用`kill gunicorn`无效，必须使用`mindinsight stop`
3. 目录结构：`summary-base-dir/xxx/profiler`

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.0.1/MindInsight/ascend/ubuntu_x86/mindinsight-1.0.1-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com

mindinsight start --summary-base-dir ./tbd00
# --port 23333
# --url-path-prefix 127.0.0.1
mindinsight stop
```

## custom operator

1. link
   * [doc-自定义算子](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/advanced_use/custom_operator.html)
2. 原语注册、算子实现、算子信息注册
3. `kernel_meta`文件有缓存帮助
4. `shape`信息改变也需要从新编译
5. GPU自定义算子
   * primitive注册，GPU kernel实现，GPU kernel注册

## auto mixed precision

1. INFO日志中的`reduce precesion`关键字
2. GPU推荐使用`level=O2`，Ascend推荐使用`level=O3`，见`ms.amp.build_train_network()`文档

## 初始化

1. `ms.common.seed.set_seed`会修改numpy随机数种子，`np.random.rand()`

## public dataset

mnist

```bash
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz
mkdir -p mnist/train
mkdir -p mnist/val
mv train-* mnist/train
mv t10k-* mnist/val
mv mnist ~/ms_data/mnist
```

cifar10

```bash
mkdir ~/ms_data
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -zvxf cifar-10-binary.tar.gz
mv cifar-10-batches-bin ~/ms_data/cifar10
```
