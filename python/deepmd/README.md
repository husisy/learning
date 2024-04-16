# deepmd

1. link
   * [documentation](https://docs.deepmodeling.org/projects/deepmd/en/master/index.html)
   * [arxiv-link](https://arxiv.org/abs/2107.02103) DP Compress: a Model Compression Scheme for Generating Efficient Deep Potential Models
2. install
   * `conda install -c conda-forge deepmd-kit lammps horovod dpdata`
3. `input.json`
   * `type_map`
   * `descriptor/sel`
4. tensorflow based

```bash
micromamba create -y -n metal-deepmd
micromamba install -y -n metal-deepmd cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm opt_einsum deepmd-kit lammps horovod dpdata
micromamba activate metal-deepmd
```

```bash
dp train input.json
# dp freeze -o graph.pb
```
