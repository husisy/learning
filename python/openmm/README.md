# openmm

1. link
   * [github](https://github.com/openmm/openmm)
   * [official-site](https://openmm.org/)
   * [documentation](http://docs.openmm.org/latest/userguide/)
2. install
   * `conda install -c conda-forge openmm openmmforcefields`
   * `python -m openmm.testInstallation`
3. application layer
4. file format
   * `pdb`
   * `PDBx/mmCIF`
   * `prmtop` (include force field, topology), `inpcrd` (include position)
   * `gro`, `top`
   * `psf`
5. conception
   * particle mesh Ewald [wiki](https://en.wikipedia.org/wiki/Ewald_summation): 适用于周期性边界条件，此时fast multiple method不是太合适
   * Amber14 force field: define parameters for proteins, DNA, RNA, lipids, water, and ions
   * TIP3P-FB water model
   * Langevin integrator: Langevin dynamics, replaces the stochastic temperature control with a velocity scaling algorithm that produces more accurate transport properties
   * friction coefficient
   * barostat：恒压
   * lone pair
   * Drude particle
   * AMOEBA force field
   * solvent溶剂, solute溶质, solute dielectric, explicit water
   * Cysteine 半胱氨酸
   * Histidine 组氨酸
   * alanine 丙氨酸
   * leucine 亮氨酸
   * amino acid 氨基酸
6. integrator
   * `VerletIntegrator`: constant energy, 和`AndersenThermostat`组合实现恒温
   * `LangevinMiddleIntegrator`: constant temperature, friction coefficient, step size, leapfrog
   * `LangevinIntegrator`: leapfrog
   * `NoseHooverIntegrator`: leapfrog
   * `BrownianIntegrator`
   * `VariableLangevinIntegrator`
   * `VariableVerletIntegrator`
7. platform: `Reference/CPU/CUDA/OpenCL`
8. environment variable
   * `OPENMM_DEFAULT_PLATFORM`
9. polarization method
   * `polarization='mutual'`
   * `polarization='extrapolated'`
   * `polarization='direct'`
10. 时间尺度
    * Planck time `1E-44s`
    * yoctosecond `1E-24s`
    * zeptosecond `1E-21s`
    * attosecond `1E-18s`
    * femtosecond `1E-15s`
    * picosecond `1E-12s`
    * nanosecond `1E-9s`
    * microsecond `1E-6s`
    * millisecond `1E-3s`
    * centisecond `1E-2s`
    * typical biomolecular force fields (AMBER/CHARMM) without constraints: `time_step<=1fs`
    * typical biomolecular force fields with `HBonds`: `time_step ~ 2fs` for verlet dynamics, `time_step ~ 4fs` for Langevin dynamics
    * `HAngles` constraint：可以更长的`time_step`
11. OpenMM默认行为：水分子是刚体`rigidWater=True`, non-rigid water需要的时间步长大约`~0.5fs`
12. coupling
    * temperature: `AndersenThermostat`
    * pressure: `MonteCarloBarostat`, `MonteCarloAnisotropicBarostat`, `MonteCarloMembraneBarostat`
13. 记录
    * 轨迹：`PDBReporter/PDBxReporter/DCDReporter`
    * `StateDataReporter`
    * `mdtraj`, `parmed`
    * `.saveState()`, `.loadState()`, `.saveCheckpoint()`, `.loadCheckpoint()`, `CheckpointReporter`
14. enhanced sampling methods: simulated tempering, metadynamics, accelerated molecular dynamics
15. design principle
    * 所有计算架构上高性能：禁止直接访问内存（GPU、分布式传输数据影响性能）
    * 易用性
    * 模块化和可拓展：force field, thermostat algorithm, new hardware
    * API硬件无关
16. 软件结构
    * public interface, OpenMM public API
    * platform independent code, OpenMM implementation layer
    * platform abstraction layer, OpenMM low level API
    * computational kernel, OpenCL/CUDA/MPI
17. public API: system, force, context, integrator, state
18. OpenMM Low Level API (OLLA): `KernelImpl`, `KernelFactory`
19. platform: `ReferencePlatform`, `CpuPlatform`, `CudaPlatform`, `OpenCLPlatform`
20. cpp unit: dalton (mass), nanometer (length), picosecond (time)
21. 头疼的单位转换 [link](http://docs.openmm.org/latest/userguide/library/05_languages_not_cpp.html#units-and-dimensional-analysis)

create force field

1. atom type
2. atom class
3. residue templates: elements and the set of bonds
4. force field parameter
5. force field XML
   * `ForceField`
   * `VirtualSite`
   * `Patch`
   * `HarmonicBondForce`

TODO 20210929 6.2.6 [link](http://docs.openmm.org/latest/userguide/application/05_creating_ffs.html#harmonicangleforce)

TODO taichi build from source

问题：taichi如果运行在单线程模式，例如超算上限制线程，那性能还剩下多少

TODO 将4个cpp-example修改为py-example
