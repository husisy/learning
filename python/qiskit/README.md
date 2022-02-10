# qiskit

1. link
   * [official site](https://qiskit.org/)
   * [github/qiskit](https://github.com/Qiskit)
   * [documentation](https://qiskit.org/documentation/index.html)
   * [github/CQuIC/qcircuit](https://github.com/CQuIC/qcircuit)
   * [github/qiskit-ionq](https://github.com/Qiskit-Partners/qiskit-ionq)
   * qiskit-documentation-applications
2. install
   * `pip install qiskit[visualization]`
   * `pip install qiskit qiskit-terra[visualization]`
   * `pip install qiskit-machine-learning sparse`
   * install `pytorch`
3. Aer, BasicAer, Terra
4. completely positive and trace-preserving (CPTP) gate
5. 对称的readout error可以转换为Kraus operator [arxiv-link-appendixB](https://arxiv.org/abs/2008.10914)
   * measurement error mitigation
   * [qiskit-documentation](https://qiskit.org/textbook/ch-quantum-hardware/measurement-error-mitigation.html)
6. relaxation time `T1`, dephasing time `T2`, `T2 < 2*T1`
7. Qiskit Ignis
8. Terra Mock Backends: real noise data for an IBM Quantum device using the date stored in Qiskit Terra
9. Quantum Approximate Optimization Algorithm (QAOA)
   * graph partition problem
10. completely positive (CP) trace-preserving: unitary
11. `optimization_level`
    * `0`: maps the circuit to the backend, with no explicit optimization (except whatever optimizations the mapper does)
    * `1`: maps the circuit, but also does light-weight optimizations by collapsing adjacent gates
    * `2`: including a noise-adaptive layout and a gate-cancellation procedure based on gate commutation relationships
    * `3`: including resynthesis of two-qubit blocks of gates in the circuit
