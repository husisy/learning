# qutip

1. link
   * [official-site](http://qutip.org/index.html)
   * [github](https://github.com/qutip/qutip)
   * [qutip-qip-documentation](https://qutip-qip.readthedocs.io/en/stable/index.html)
   * [github/qutip-notebook](https://github.com/qutip/qutip-notebooks)
2. install
   * `conda install -c conda-forge qutip`
   * `pip install qutip`
   * `pip install qutip-qip`
3. SNOT: Hadamard gate
4. isolated quantum system
   * unitary time evolution
   * non-controllable drist Hamiltontian $H_d$, controllable drive Hamiltonian $H_c$
5. open quantum system
   * Lindblad master equation
   * sampling Monte Carlo trajectories
6. pulse-level simulation
   * drive strength
   * interaction strength
   * decoherence noise: T1, T2
   * cubic spline interpolation
   * `tlist`: time sequence for intermediate results
   * `e_ops`: measurement observables
   * `ModelProcessor`: spin chain, superconducting qubits, dispersive cavity QED
   * compiler
   * scheduler
   * noise: noise in hardware model, control noise, Lindblad noise
7. `SCQubits`
   * three-level system: ground state, the first excited state
   * the leakage of the population
   * orthogonal quadratures $a+a^\dagger$, $i(a-a^\dagger )$
   * adjacent ineraction: cross resonant pulse
   * parameter references
     * Effective hamiltonian models of the cross-resonance gate [doi-link](https://doi.org/10.1103/PhysRevA.101.052308)
     * Circuit quantum electrodynamics [doi-link](https://doi.org/10.1103/RevModPhys.93.025005)
8. `DispersiveCavityQED`
   * system: multi-level cavity, qubit system
   * qubit-cavity interaction $a^\dagger \sigma^- + a\sigma^+$
   * device parameter: cavity frequency, qubit frequency, detuning, interaction strength
9. `OptPulseProcessor`
