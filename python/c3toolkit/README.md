# c3toolkit

1. link
   * [github/c3](https://github.com/q-optimize/c3)
   * [documentation](https://c3-toolset.readthedocs.io/en/master/)
   * [arxiv-link](https://arxiv.org/abs/2009.09866) Integrated tool-set for Control, Calibration and Characterization of quantum devices applied to superconducting qubits
   * [arxiv-link](https://arxiv.org/abs/2205.04829) Software tool-set for automated quantum system identification and device bring up
   * [arxiv-link](https://arxiv.org/abs/1809.04919) Counteracting systems of diabaticities using DRAG controls: The status after 10 years
2. install
   * `pip install c3-toolset-nightly "qiskit[visualization]"`
   * `pip install c3-toolset "qiskit[visualization]"`
   * `python/conda/README-mamba.md:env-tf`
3. concept
   * C1: open-loop optimal control, given a model, find the pulse shapes which maximize fidelity with a target operation
   * C2: closed-loop calibration, given pulses, calibrate their parameters to maximize a figure of merit measured by the actual experiment
   * C3: model learning, given control pulses and their experimental measurement outcome, optimize model parameters to best reproduce the results
   * transmon, duffing oscillator
   * `.set_lindbladian()` coherent or open-system dynamics
   * `.set_dressed()` whether to eliminate the static coupling by going to the dressed frame
   * envelope signal, arbitrary waveform generator (AWG), local oscillator (LO)
4. limitation
   * no apple silicon support
