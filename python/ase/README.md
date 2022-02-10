# Atomic Simulation Environment

1. link
   * [documentation](https://wiki.fysik.dtu.dk/ase/)
   * [gitlab](https://gitlab.com/ase/ase)
   * [supported-calculators](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html#supported-calculators)
2. 安装 `pip install ase`
3. abbreviation
   * projector-augmented wave (PAW)
   * effective medium theory (EMT)
   * linear combination of atomic orbitals (LCAO)
4. file format
   * `.xyz`: Simple xyz-format
   * `.cube`: Gaussian cube file
   * `.pdb`: Protein data bank file
   * `.traj`: ASE’s own trajectory format
5. command line utility
   * `ase gui xxx.traj`
   * `ase info --formats`
   * `ase gui -r 3,3,2 xxx.xyz`: repeat
   * `ase info --calculators`
6. G2 molecule dataset

## gpaw

1. link
   * [documentation](https://wiki.fysik.dtu.dk/gpaw/index.html)
   * [gitlab](https://gitlab.com/gpaw/gpaw)
2. install
   * `conda install -c conda-forge gpaw`
   * `pip` fail for conda-env
3. command line utility
   * `gpaw info`
   * `gpaw test`
4. `gpaw.txt`
5. mode
   * `fd` real-space grids
   * `pw` planewaves
   * `lcao` localized atomic orbitals
6. basis
   * `dzp` the standard double-zeta polarized basis set

## sisl

1. link
   * [github](https://github.com/zerothi/sisl)

TODO

1. [ ] siesta
2. [ ] TranSiesta
3. [ ] TBtrans
