# pytorch extension

1. link
   * [pytorch / notes / extending](https://pytorch.org/docs/master/notes/extending.html)
   * [pytorch / tutorials / custom cpp cuda extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
   * [github / NNPACK](https://github.com/Maratyszcza/NNPACK)
   * [github / ATen](https://github.com/zdevito/ATen/tree/master/cmake)
   * [pytorch/documentation/cpp-api](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html)
   * [github/Yanqi-Chen/spikingjellyCppExt](https://github.com/Yanqi-Chen/spikingjellyCppExt)
2. **必须**先`import torch`，在执行自定义包的导入, resolve some symbols that the dynamic linker must see

## mwe00 (ws00)

`setuptools`

1. 安装 `pip install .`
2. 卸载 `pip uninstall my_pytorch_ext`
3. 运行示例 `python draft00.py`

just-in-time (JIT)

1. 建议先卸载通过`setuptools`方式安装的包 `pip unisntall my_pytorch_ext`
2. 运行示例 `python draft01.py`

## mwe01 (ws01)

[link](https://github.com/pytorch/extension-cpp)

[pytorch/documentation/custom-cpp-and-cuda-extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)

```bash
export CUDA_HOME=/usr/local/cuda-11.3
```

1. 安装 `pip install .`
2. 卸载 `pip uninstall my_pytorch_ext`
3. 运行示例 `python draft00.py`
