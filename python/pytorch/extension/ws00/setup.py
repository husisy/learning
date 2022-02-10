import setuptools
import torch
import torch.utils.cpp_extension

# from torch.utils import cpp_extension
# setuptools.Extension

setuptools.setup(name='my_pytorch_ext',
      ext_modules=[torch.utils.cpp_extension.CppExtension('my_pytorch_ext', ['lltm.cpp'])],
      cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension})
