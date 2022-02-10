import setuptools
import torch.utils.cpp_extension
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension

tmp0 = [
    torch.utils.cpp_extension.CUDAExtension('my_pytorch_ext', ['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'])
]
setuptools.setup(
    name='my_pytorch_ext',
    ext_modules=tmp0,
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension},
)
