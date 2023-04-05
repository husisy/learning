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

micromamba create -y -n cuda117
micromamba install -y -n cuda117 -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.7
micromamba install -y -n cuda117 cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum cupy nccl

micromamba create -y -n metal
micromamba install -y -n metal -c pytorch pytorch torchvision torchaudio
micromamba install -y -n metal cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum cvxpy scs
```
