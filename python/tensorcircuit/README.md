# tensorcircuit

1. link
   * [github](https://github.com/tencent-quantum-lab/tensorcircuit)
   * [documentation](https://tensorcircuit.readthedocs.io/)
2. install
   * `pip install tensorcircuit`
3. backend
   * `jax`, `tensorflow`, `torch`
   * backend agnosticism
4. contractor: `opt_einsum`, `cotengra`, `kahypar`

```bash
# https://stackoverflow.com/a/73420441
pip uninstall h5py
pip install --no-cache-dir h5py

pip install "tensorcircuit[torch,jax]"
pip install "tensorcircuit[tensorflow]"
pip install optax
```
