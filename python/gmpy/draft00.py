import gmpy2
from gmpy2 import mpz,mpq,mpfr,mpc

gmpy2.set_context(gmpy2.context())
# gmpy2.get_context().precision=200

## mpz: arbitrary precision integer
mpz(99)*43 #mpz(4257)
pow(mpz(99), 37, mod=59) #mpz(18)
gmpy2.isqrt(99) #mpz(9)
gmpy2.isqrt_rem(99) #mpz(9), mpz(18)
gmpy2.gcd(123, 27) #mpz(3)
gmpy2.lcm(123,27) #mpz(1107)
(mpz(123) + 12)//5 #mpz(27)
(mpz(123) + 12)/5 #mpfr(27.0)
(mpz(123) + 12)/5.0 #mpfr(27.0)


## mpq: arbitrary precision rational
mpq(3, 7)/7 #mpq(3, 49)
mpq(45,3) * mpq(11,8) #mpq(165, 8)


## mpfr: arbitrary precision floating point
mpfr(1)/7
with gmpy2.local_context() as ctx:
    print(gmpy2.const_pi())
    ctx.precision += 20
    print(gmpy2.const_pi())


## mpc: arbitrary precision complex
with gmpy2.local_context() as ctx:
    ctx.allow_complex = True #default=False
    x0 = gmpy2.sqrt(mpfr(-1))


## misc
gmpy2.license()
gmpy2.mp_version()
gmpy2.mpc_version()
gmpy2.mpfr_version()
gmpy2.version()
