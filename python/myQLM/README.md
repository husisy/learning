# myQLM

1. link
   * [documentation](https://myqlm.github.io/index.html)
   * [github](https://github.com/myQLM)
2. install
   * `pip install myqlm`
   * `micromamba install -c myqlm myqlm`
3. quantum paradigms
   * gate-based paradigm
   * the analog paradigm
   * the quantum annealing paradigm

```bash
#linux only, failed on macOS
micromamba create -y -n myqlm
micromamba install -y -n myqlm cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum
micromamba activate myqlm
pip install myqlm
```
