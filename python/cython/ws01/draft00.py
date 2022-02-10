import my_cython_pkg
import my_cython_cpp_pkg

my_cython_pkg.hello('233')

my_cython_pkg.demo_integrate(-1, 1, 233)

my_cython_pkg.primes(100)

my_cython_cpp_pkg.prime_cvector(100)

my_cython_pkg.demo_libc_stdlib_atoi(b'233')

my_cython_pkg.demo_libc_math_sin(0.233)
