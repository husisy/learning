# sympy

1. reference
   * [official site](https://www.sympy.org/en/index.html)
   * [github](https://github.com/sympy/sympy)
   * [get started](https://docs.sympy.org/latest/tutorial/index.html)
   * [Sympy Live](http://live.sympy.org/)
2. install
   * conda: `conda install -c conda-forge sympy`
3. feature: free, Python-based, lightweight
4. concept
   * Computer Algebra Systems (CASs)

```Python
import sympy
x = sympy.symbols('x')
sympy.limit(sympy.sin(x)/x, x, 0)
sympy.integrate(1/x, x)
```
