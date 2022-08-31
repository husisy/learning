# QuTech

1. link
   * [github/cqc](https://github.com/SoftwareQuTech/CQC-Python)
   * [cqc/documentation](https://softwarequtech.github.io/CQC-Python/index.html)
   * [github/SimulaQron](https://github.com/SoftwareQuTech/SimulaQron)
2. cqc is deprecated in favor of netqasm

```bash
conda create -y -n cuda113-cqc
conda install -y -n cuda113-cqc -c conda-forge cudatoolkit=11.3
conda install -y -n cuda113-cqc -c pytorch pytorch torchvision torchaudio
conda install -y -n cuda113-cqc -c conda-forge cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum cupy bitstring
# necessary for cqc: bitstring
# conda install -n cuda113-cqc -c conda-forge anytree bitstring twisted
conda activate cuda113-cqc
pip install netqasm simulaqron "numpy>=1.22" "click>8"
```
