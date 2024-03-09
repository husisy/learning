# qctrl

1. link
   * [documentation](https://docs.q-ctrl.com/boulder-opal)
   * [github/open-controls](https://github.com/qctrl/open-controls)
   * unread
     * [qctrl-tutorial](https://docs.q-ctrl.com/boulder-opal/application-notes/performing-narrow-band-magnetic-field-spectroscopy-with-nv-centers#spectrum-reconstruction-with-boulder-opal)
2. concept
   * quantum computing
   * quantum sensing
   * friend companys: [quantum-machines](https://www.quantum-machines.co/products/opx/) [artiq](https://m-labs.hk/)
3. install
   * `pip install qctrl qctrl-open-controls qctrl-visualizer qctrl-qua`
   * `qctrl auth` (fail to authenticate on vscode remote)
4. quantum control engineering
   * characterize hardware: identify key system parameters and imperfections for effective calibration, simulation and optimization
   * design error-robust controls: create control solutions to manipulate quantum systems that are resilient to noise and errors
   * simulate quantum dynamics: understand the anticipate the behavior of complex quantum devices under realistic conditions
   * automate hardware with AI: automate and speed up calibration and optimization with closed loop agents at scale
   * verify performance: evaluate control solutions to gain insights and ensure effectiveness
5. qctrl concept
   * tensorflow-based (graph-based optimization engine)
   * boulder opal
   * PWC: piecewise-constant function
   * CORPSE: Creates a compensating for off-resonance with a pulse sequence
   * GKSL equation: Gorini–Kossakowski–Sudarshan–Lindblad equation
6. estimate Hamiltonian model parameters
   * mean squared error loss
   * covariance matrix: inverse of the Hessian matrix, estimate the precision
   * confidense region [wiki](https://en.wikipedia.org/wiki/Confidence_region)
   * Cramér–Rao bound
   * [PRApplied2020-doi](https://doi.org/10.1103/PhysRevApplied.14.024021) Simultaneous Spectral Estimation of Dephasing and Amplitude Noise on a Qubit Sensor via Optimally Band-Limited Control
7. control-design optimization strategy
   * model of system to optimize control offline, uncertainty in a mathematica lmodel
   * no noise, weak noise, strong noise
   * few parameter or high-dimensional parameter
   * [arxiv-link](https://arxiv.org/abs/2001.04060) Software tools for quantum control: Improving quantum computer performance through noise and error suppression
   * control strategy
     * robust control `noise < 1%` [PRApplied-link](https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.15.064054) [qctrl-tutorial](https://docs.q-ctrl.com/boulder-opal/tutorials/design-robust-single-qubit-gates-using-computational-graphs)
     * the first-order Magnus approximation of the toggling frame Hamiltonian
     * the second-order approximation of the time evolution operator
     * stochastic optimization for strong noise sources [qctrl-tutorial](https://docs.q-ctrl.com/boulder-opal/user-guides/how-to-optimize-controls-robust-to-strong-noise-sources)
     * closed-loop hardware optimization
       * covariance matrix adaptation evolution strategy (CMA-ES)
       * Gaussian processes
       * simulated annealing
8. AI automation
   * [blog-link](https://q-ctrl.com/blog/firing-up-quantum-algorithms-boosting-performance-up-to-9000x) [PRA2023-link](https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.20.024034)
   * [webinar-link](https://q-ctrl.com/webinars/automated-closed-loop-hardware-optimization) automated closed-loop hardware optimization
   * difficulty: the number of parameters grows exponentially with the number of qubits, nonlinearly coupled, unknowable system parameters
   * ML-enhanced closed-loop hardware optimization
     * no need for physical model
     * robust to noise and model uncertainty
     * experimentally efficient
     * novel control strategies
   * M-Loop [documentation](https://m-loop.readthedocs.io/en/stable/api/mloop.html) [github](https://github.com/michaelhush/M-LOOP)
   * [PRX-quantum2021-doi](https://doi.org/10.1103/PRXQuantum.2.040324) Experimental Deep Reinforcement Learning for Error-Robust Gate-Set Design on a Superconducting Quantum Computer
9. QUA from company "Quantum Machine" [website](https://www.quantum-machines.co/)
   * [PRL2020-doi](https://doi.org/10.1103/PhysRevLett.125.170502) High-Fidelity Software-Defined Quantum Logic on a Superconducting Qudit
   * operator-X (OPX) quantum devices
   * hanning function for smoothing [wiki-link](https://en.wikipedia.org/wiki/Hann_function)
10. verify performance
    * quasi-static scans
    * susceptibility of a control to two different noise processes applied simultaneously
      * time-dependent multiplicative noise on a drive $\beta$
      * the additive dephasing noise which acts independent of any controls $\eta$

toy Hamiltonian

$$ H=\alpha\Omega(t)\sum_{k}\sigma_{x}^{(k)}+\delta\sum_{k}\sigma_{z}^{\left(k\right)}+\gamma\bigotimes_{k}\sigma_{z}^{\left(k\right)} $$

1. $\sigma_i^{(k)}$: the $i$ Pauli matrix acting on the k-th qubit
2. $\alpha\Omega(t)$: Rabi coupling due to external pulse
   * unitless input value $\Omega(t)$
   * scaling factor relating these to the physical coupling occurring in the system $\alpha$
3. $\delta$: detuning of the qubit
4. $\gamma$: the interaction strength between qubits

estimate the bandwidth of transmission line via probe measurement [link](https://docs.q-ctrl.com/boulder-opal/user-guides/how-to-characterize-a-transmission-line-using-a-qubit-as-a-probe)

design robust single-qubit gates using computational graph [link](https://docs.q-ctrl.com/boulder-opal/tutorials/design-robust-single-qubit-gates-using-computational-graphs)

$$ H(t)=\alpha(t)\sigma_{z}+\frac{1}{2}\left(\gamma(t)\sigma_{-}+\gamma^{*}(t)\sigma_{+}\right)+\delta\sigma_{z}+\beta(t)\sigma_{z} $$

1. parameter
   * $\alpha(t)$: real-valued time-dependent control
   * $\gamma(t)$: complex-valued time-dependent control
   * $\delta$: detuning
   * $\beta(t)$: dephasing noise process
2. a qubit coupled to a classical bath
   * $\beta(t)$ is slowly varying so that we can assume that it is constant at each different realization

## short-note

### characterize hardware

A device is characterized by a Hamiltonian with time-dependent unknown parameters, e.g.

$$ H(\omega)=\omega(t)X+Z $$

and the target is to determine these parameters. Try to get some data $\left(|\psi_{i}\rangle,A,a,t\right)$, where $|\psi_{i}\rangle$ is the initial state and the measurement result a is taken after some time duration $t$

$$ a=\langle\psi_{i}|e^{iHt}Ae^{-iHt}|\psi_{i}\rangle+\eta $$

with noise $\eta$. To do this task, let's minimize the following loss function

$$ \mathcal{L}\left(\theta\right)=\sum_{s}\left(\langle\psi_{i}|e^{iH(\theta)t}Ae^{-iH(\theta)t}|\psi_{i}\rangle-a\right)^{2} $$

where $s$ labels the $s$-th data sample.

### verify performance

Susceptibility of a control pulse

[qctrl-tutorial00](https://docs.q-ctrl.com/boulder-opal/user-guides/how-to-evaluate-control-susceptibility-to-quasi-static-noise) given a optimized control pulse $\omega(t)$ to implement a target gate $\hat{U}$, denote the system Hamiltonian without noise as $H(\omega(t))$. The susceptibility is to analyze the effect when the Hamiltonian is suffered from various noise. Two common types are

1. the time-dependent multiplicative noise on a drive $\beta$
2. the additive dephasing noise which acts independent of any controls $\eta$, e.g. dephasing noise

The "effect" could be how the gate infidelity changes at different scale of noise.

$$ H(\beta,\eta)=(1+\beta)H(\omega)+\eta Z $$

$$ F(A,B)=\left|\mathrm{Tr}[A^{\dagger}B]\right|^{2}/d^{2} $$

$$ 1-F\left(\hat{U},U\left[H(\beta,\eta)\right]\right) $$

The "analysis" could be

1. visualize the infidelity landscape with respect to the noise parameters
2. some ctrl pulse is more robust to one kind noise, but more sensitive to another kind noise
3. for some fixed initial state, evolved under some Lindblad master equation (e.g. open system dynamics), then analyze the infidelity of the final density matrix with respect to the noise parameters
4. filter function [qctrl-tutorial](https://docs.q-ctrl.com/boulder-opal/user-guides/how-to-calculate-and-use-filter-functions-for-arbitrary-controls)
5. various single-qubit control sequence is predefined in [qctrl-open-ctrl](https://docs.q-ctrl.com/open-controls/references/qctrl-open-controls/qctrlopencontrols/new_bb1_control.html) package, based on them, qctrl can compare their performance.
   * primitive: sensitive to both dephasing and amplitude noise
   * BB1: suppress amplitude noise
   * CORPSE: suppress dephasing noise
   * CORPSE in BB1: suppress both noise
