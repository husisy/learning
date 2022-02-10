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

## space

1. Euclidean space
   * [wiki](https://en.wikipedia.org/wiki/Euclidean_space)
   * Euclidea vector space: a finite-dimensional inner product space over the real numbers
2. vector space
   * [wiki](https://en.wikipedia.org/wiki/Vector_space)
   * a vector space over a field $F$ is a set $V$ together with two operatos that satisfy the eight axioms
   * operation: addition, scalar multiplication
   * 加法结合律 associativity of addition `u+(v+w)=(u+v)+w`
   * 加法交换律 commutativity of addition `u+v=v+u`
   * 加法零元 identity element of addition `v+0=v`
   * 加法逆 inverse elements of addition `v+(-v)=0`
   * 数乘结合律 compatibility of scalar multiplication `a(bu)=(ab)u`
   * 数乘单位元 identity element of scalar multiplication `1*u=u`
   * 数乘矢量分配律 distributivity of scalar multiplication with respect to vector `a(u+v)=au+av`
   * 数乘标量分配律 distributivity of scalar multiplication with respect to scalar `(a+b)u=au+bu`
3. inner product space
   * [wiki](https://en.wikipedia.org/wiki/Inner_product_space)
   * an inner product space is a vector space $V$ over the field $F$ together with an inner product
   * conjugate symmetry `<x,y>=<y,x>^H`
   * linearity in the first argument `<ax,y>=a<x,y>` `<x+y,z>=<x,z>+<y,z>`
   * positive-definite
4. metric space
   * [wiki](https://en.wikipedia.org/wiki/Metric_space)
   * a metric space is an ordered pair $(M,d)$ where $M$ is a set and $d$ is a metric (distance function)
   * identity of indisernibles `d(x,y)=0 <=> x=y`
   * symmetry `d(x,y)=d(y,x)`
   * subadditivity / triangle inequality `d(x,z) <= d(x,y) + d(y,z)`
   * complete space
   * compact space
5. topological space
   * [wiki](https://en.wikipedia.org/wiki/Topological_space)
6. Hilbert space
   * [wiki](https://en.wikipedia.org/wiki/Hilbert_space)
   * a Hilbert space $H$ is a real or complex inner produce space that is also a complete metric space with respect to the distance function induced by the inner product
   * pre-Hilbert space：内积空间 + metric space
   * Hilbert space：内积空间 + complete metric space；一个incomplete metric space的栗子见[wiki](https://en.wikipedia.org/wiki/Inner_product_space#Hilbert_space)
   * distance function (norm): symmetric in $x$ and $y$, nonzero, triangle inequality (Cauchy-Schwarz inequality)
