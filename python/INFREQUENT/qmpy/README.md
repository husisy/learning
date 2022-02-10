# OQMD

mysql数据库太大`150-200GB`，完全下载并在本地host不现实，而且qmpy库依赖过多，极其不好安装

```bash
# fail
conda create -n oqmd
conda install -n oqmd -c conda-forge cython ipython matplotlib h5py pandas pylint jupyter jupyterlab pillow protobuf scipy requests tqdm lxml joblib scikit-learn seaborn scikit-image mysql-connector-python django pyyaml django-extensions pulp
conda activate oqmd
pip install python-memcached
pip install qmpy
pip install tensorflow
```

1. RESTful API: `pip install qmpy-rester`
2. [github-qmpy-rester](https://github.com/mohanliu/qmpy_rester)
3. attributes
   * `composition`: compostion of the materials or phase space, e.g. Al2O3, Fe-O
   * `element_set`: the set of elements that the compound must have, '-' for OR, ',' for AND, e.g. (Fe-Mn),O
   * `icsd`: whether the structure exists in ICSD, e.g. False, True, F, T
   * `prototype`: structure prototype of that compound, e.g. Cu, CsCl
   * `generic`: chemical formula abstract, e.g. AB, AB2
   * `spacegroup`: the space group of the structure, e.g. Fm-3m
   * `natoms`: number of atoms in the supercell, e.g. 2, >5
   * `volume`: volume of the supercell, e.g. >10
   * `ntypes`: number of elements types in the compound, e.g. 2, <3
   * `stability`: hull distance of the compound, e.g. 0, <-0.1,
   * `delta_e`: formation energy of that compound, e.g. <-0.5,
   * `band_gap`: band gap of the materials, e.g. 0, >2
   * `fields`: return subset of fields, e.g. 'name,id,delta_e', '!sites'
   * `filter`: customized filters, e.g. 'element_set=O AND ( stability<-0.1 OR delta_e<-0.5 )'
   * `limit`: number of data return at once
   * `offset`: the offset of data return

```bash
conda create -n oqmd
conda install -n oqmd -c conda-forge autopep8 cython ipython pytest matplotlib h5py pandas pylint jupyter jupyterlab pillow protobuf scipy requests tqdm lxml
conda activate oqmd
pip install qmpy-rester
```
