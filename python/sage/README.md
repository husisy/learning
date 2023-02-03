# sage

1. link
   * official-site
   * [github-mirror](https://github.com/sagemath/sage/)
   * [documentation](https://doc.sagemath.org/html/en/index.html)

```bash
conda create -y -n sage
# conda install -y -n sage -c conda-forge cudatoolkit=11.7
# conda install -y -n sage -c pytorch pytorch torchvision torchaudio
conda install -y -n sage -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.7 python=3.9
conda install -y -n sage -c conda-forge sage cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum cupy

```
