# Cirq

1. link
   * [documentation](https://cirq.readthedocs.io/en/stable/)
   * [github](https://github.com/quantumlib/Cirq)
2. install
   * `pip install cirq`
   * `sudo apt install texlive-latex-base latexmk`
   * test `python -c 'import cirq_google; print(cirq_google.Foxtail)'`
3. noise channel, see John Preskill's note
4. to run unittest, first `cp ../my_quantum_circuit/np_quantum_circuit.py .`

TODO

1. build unittest: test wave amplitude, random initial value, test gate, test-symbol

## example: variational quantum algorithm

1. link
   * variational method, [arxiv:1304.3061](https://arxiv.org/abs/1304.3061), [arxiv:1507.08969](https://arxiv.org/abs/1507.08969)
2. concept
   * ansatz state: functioin of some parameter
   * Quantum Approximate Optimization Algorithm (QAOA)
