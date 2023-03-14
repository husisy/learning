# galois

1. link
   * [github](https://github.com/mhostetter/galois)
   * [documentation](https://mhostetter.github.io/galois/latest/)
2. install `pip install galois`
3. compilation mode
   * `jit-lookup`: exponential and logarithm lookup table $\log_\alpha(x)=i$, Zech's logarithm: $Z_\alpha(x)=\log_\alpha(1+\alpha^x)$
   * `jit-calculate`
   * `python-claculate`
4. representation
   * integer `int`
   * polynomial `poly`
   * power `power`
   * vector
5. prime field $GF(p)$: $\left\{ 0,1,\cdots,p-1 \right\}$
6. extension field $GF(p^m)$: degree $m-1$ polynomials over $GF(p)$, $GF(p)[x]/f(x)$
7. Bezout identity [wiki](https://en.wikipedia.org/wiki/B%C3%A9zout%27s_identity)
8. Extended Euclidean algorithm [wiki](https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm)
9. Conway polynomial $C_{p,m}(x)$
