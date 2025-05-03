# Mamba

1. link
   * [github](https://github.com/mamba-org/mamba)
   * [documentation](https://mamba.readthedocs.io/en/latest/)
   * [github/miniforge](https://github.com/conda-forge/miniforge#mambaforge)
2. mamba, micromamba, libmamba
3. install micromamba
   * linux: `curl micro.mamba.pm/install.sh | bash`
   * macOS: `curl micro.mamba.pm/install.sh | zsh`
4. prefix/environment
   * root prefix
5. 个人偏见
   * 将`conda-forge`设置为默认channel
6. `.mambarc`
7. pip mirror sites [link](https://mirrors.cernet.edu.cn/about)

```bash
$PREFIX
├── include
├── lib
├── lib[arch]
├── bin
├── etc
└── share
```

```yaml
# .mambarc
channels:
  - conda-forge
always_yes: false
```

```bash
micromamba create -n test00
micromamba install -n test00 cython matplotlib h5py pandas pillow protobuf scipy requests tqdm flask ipython openai python-dotenv
micromamba env list
micromamba activate test00
micromamba self-update

micromamba repoquery search python-dotenv
micromamba repoquery depends python-dotenv
micromamba repoquery depends --recursive conda-forge python-dotenv

micromamba create -y -n nocuda
micromamba install -y -n nocuda pytorch cython ipython pytest matplotlib h5py pandas pillow protobuf scipy requests tqdm lxml opt_einsum

micromamba create -y -n cuda118
micromamba install -y -n cuda118 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
micromamba install -y -n cuda118 python cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum cupy nccl cvxpy pytest-xdist pytest-cov
micromamba install -y -n cuda118 -c MOSEK MOSEK
pip install --force-reinstall "numpy>=1.26" #conda-forge/numpy is broken

micromamba create -y -n cuda128
micromamba install -y -n cuda128 python cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum cvxpy pytest-xdist pytest-cov torch
micromamba install -y -n cuda128 -c MOSEK MOSEK
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -U --force-reinstall numpy scipy cvxpy

micromamba create -y -n cuda126
micromamba install -y -n cuda126 python=3.12 cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum cvxpy pytest-xdist pytest-cov pytorch
micromamba install -y -n cuda126 -c MOSEK MOSEK

# x86-64 cpu environment for tensorflow
micromamba create -n env-tf
micromamba install -y -n env-tf python=3.11 cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum cvxpy scs pytest-xdist
micromamba activate env-tf
pip install tensorflow

# x86-64 cuda118 tensorflow
micromamba create -y -n cuda118-tf
micromamba install -y -n cuda118-tf cudatoolkit=11.8 cudnn python=3.11 cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum cvxpy scs
micromamba activate cuda118-tf
pip install tensorflow
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/zhangc/micromamba/envs/cuda118-tf/lib"

micromamba create -y -n metal
micromamba install -y -n metal cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum cvxpy scs pytest-xdist pytest-cov seaborn
micromamba install -y -n metal -c MOSEK MOSEK
micromamba install -y -n metal -c pytorch pytorch torchvision #conda-forge/macOS/pytorch is broken
micromamba activate metal
# pip install torch torchvision #conda-forge/macOS/pytorch is broken

micromamba create -y -n metal311
micromamba install -y -n metal311 python=3.11 cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum cvxpy scs pytest-xdist pytest-cov
micromamba install -y -n metal311 -c MOSEK MOSEK
micromamba install -y -n metal311 -c pytorch pytorch torchvision #conda-forge/macOS/pytorch is broken
micromamba activate metal311
pip install torchrl

micromamba create -y -n metal310
micromamba install -y -n metal310 python=3.10 cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum cvxpy scs pytest-xdist pytest-cov pennylane docplex qiskit qiskit-terra dwave-system
# pennylane-lightning
micromamba install -y -n metal310 -c MOSEK MOSEK
micromamba install -y -n metal310 -c pytorch pytorch torchvision #conda-forge/macOS/pytorch is broken
micromamba activate metal310
pip install openqaoa

micromamba create -y -n metal-tf
micromamba install -y -n metal-tf cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm opt_einsum
micromamba activate metal-tf
pip install tensorflow


# install numpy>=1.26
micromamba create -y -n metal-acc
micromamba install -y -n metal-acc "libblas=*=*accelerate" cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum cvxpy scs pytest-xdist pytest-cov
micromamba install -y -n metal-acc -c MOSEK MOSEK
micromamba activate metal-acc
pip install --force-reinstall "numpy>=1.26" #conda-forge/numpy is broken
pip install torch torchvision #conda-forge/macOS/pytorch is broken
```
