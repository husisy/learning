# PySCF

Python-based simulations of chemistry framework

1. link
   * [documentation](https://sunqm.github.io/pyscf/)
   * [github](https://github.com/sunqm/pyscf)
2. `~/.pyscf_conf.py`, `pyscf.__config__`
   * 环境变量`PYSCF_MAX_MEMORY`, `PYSCF_CONFIG_FILE`
3. 安装
   * `pip install pyscf`, `pip install pyscf[geomopt]`
   * `conda install -c pyscf pyscf`
4. type
   * Restricted Open-shell Hartree-Fock (ROHF)
   * Restricted Hartree-Fock (RHF)
   * Unrestricted Hartree-Fock (UHF)
   * [link](https://www.tau.ac.il/~ephraim/RHF.pdf)
   * Restricted Kohn-Sham (RKS)
5. Gaussian type orbitals (GTO)
   * [link](https://www.theochem.ru.nl/~pwormer/Knowino/knowino.org/wiki/Gaussian_type_orbitals.html)
   * cartesian GTO, spherical GTO
   * contracted sets of primitive GTOs
   * Gaussian product rule
6. quantum mechanics / molecular mechanics (QM/MM)
   * [wiki-QMMM](https://en.wikipedia.org/wiki/QM/MM)
7. unit system
   * length: `angstrom` (default), `Bohr`
8. concept
   * Effective core potentials (ECP)
   * Spin-orbit (SO) ECP
   * Slater-type orbitals (STO)
   * Algebraic Diagrammatic Construction (ADC)
   * atomic orbital (AO)
   * symmetry-adapted orbital (SO)
   * direct inversion in the iterative subspace (DIIS)
   * second-order SCF (SOSCF)
   * co-iterative augmented hessian (CIAH)
   * spin-free eXact-2-component (SFX2C)
9. DFT integration grid
   * Bragg radius for atom
   * Treutler-Ahlrichs radial grids
   * Becke partition for grid weights
   * NWChem pruning scheme
   * mesh grids
   * elements/radial/angular: `H,He/50/302`, `Li-Ne/75/302`, `Na-Ar/80/434`, `K-Kr/90/434`, `Rb-Xe/95/434`, `Cs-Rn/100/434`
