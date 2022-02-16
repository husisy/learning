# qutip

1. link
   * [official-site](http://qutip.org/index.html)
   * [github](https://github.com/qutip/qutip)
2. bra / ket / density matrix

## Optimal control

1. link
   * [qutip-overview-optimal-control](https://nbviewer.jupyter.org/github/qutip/qutip-notebooks/blob/master/examples/optimal-control-overview.ipynb)
2. keyword
   * time-independent dynamics generator (drift dynamics generator, drift Hamiltonian)
   * controlllability, quantum optimal control
   * Lie Algebra Rank Criterion, see [Introduction-to-Quantum-Control-and-Dynamics](https://www.crcpress.com/Introduction-to-Quantum-Control-and-Dynamics/DAlessandro/p/book/9781584888840)
   * gate synthesis
3. GRadient Ascent Pulse Engineering (GRAPE)
   * [PubMed-Optimal-control-of-coupled-spin-dynamics:-design-of-NMR-pulse-sequences-by-gradient-ascent-algorithms](https://www.ncbi.nlm.nih.gov/pubmed/15649756)
   * piecewise constant approximation
   * Broyden-Fletcher-Goldfarb-Shanno Algorithm (BFGS), L-BFGS-B
   * Frechet derivative method
4. Chopped RAndom Basis (CRAB) algorithm
   * the dimension of a quantum optimal control problem is a polynomial function of the dimension of the manifold of the time-polynomial reachable states, when allowing for a finite control precision and evolution time
   * [PRA-Chopped-random-basis-quantum-optimization](https://doi.org/10.1103/PhysRevA.84.022326)
   * escape from local minima: [PRA-Dressing-the-chopped-random-basis-optimization:-A-bandwidth-limited-access-to-the-trap-free-landscape](https://doi.org/10.1103/PhysRevA.92.062343)
5. Kraus, Liouville supermatrix and Choi matrix formalisms
   * Havel, T. Robust procedures for converting among Lindblad, Kraus and matrix representations of quantum dynamical semigroups. Journal of Mathematical Physics 44 2, 534 (2003) [paper](http://dx.doi.org/10.1063/1.1518555)
   * Watrous, J. Theory of Quantum Information, [lecture-notes](https://cs.uwaterloo.ca/~watrous/CS766/)
   * Liouvillian superoperator
   * Hinton diagram
6. Choi matrix
   * ancilla-assisted process tomography (AAPT)
7. Kraus representation
8. Stinespring representation
9. quantum maps
   * positive quantum map
   * completely positive quantum map
