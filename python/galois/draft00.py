import numpy as np

import galois


GF3x5 = galois.GF(3**5)
# for speed up
GF3x5 = galois.GF(3, 5)
GF109987x4 = galois.GF(109987, 4, irreducible_poly="x^4 + 3x^2 + 100525x + 3", primitive_element="x", verify=False)
GF3x5.properties #str
GF3x5.name
GF3x5.characteristic
GF3x5.degree
GF3x5.order
GF3x5.irreducible_poly #Poly(x^5 + 2x + 1, GF(3))
GF3x5.is_prime_field
GF3x5.is_primitive_poly
GF3x5.primitive_element
GF3x5.element_repr #int

# galois.Array
issubclass(GF3x5, galois.FieldArray) #True
issubclass(GF3x5, np.ndarray) #True

gf0 = GF3x5([236,87,38,112])
isinstance(gf0, galois.FieldArray) #True
isinstance(gf0, np.ndarray) #True
type(gf0) is GF3x5

np0 = np.array([236,87,38,112], dtype=np.int64)
gf1 = np0.view(GF3x5)
np1 = gf1.view(np.ndarray)
gf2 = GF3x5.Random((3,2), seed=1)
gf3 = GF3x5.Identity(4)

gf0.multiplicative_order() #[236, 87, 38, 112]


# representation: integer(int) polynomial(poly) power(power)
GF3x5.repr('poly')
GF3x5.repr('power')
GF3x5.repr('int')
GF3x5_i = galois.GF(3**5, repr='int')
GF3x5_pl = galois.GF(3**5, repr='poly')
GF3x5_pw = galois.GF(3**5, repr='power')
with GF3x5.repr('poly'):
    pass

# arithmetic
gf0 = GF3x5([236, 87, 38, 112])
gf1 = GF3x5([109, 17, 108, 224])
gf0 + gf1 #[18, 95, 146, 0]
gf0 - gf1 #[127, 100, 173, 224]
gf0 * gf1 #[21, 241, 179, 82]
gf0 / gf1 #[67, 47, 192, 2]
np.sqrt(gf0) #[51, 135, 40, 16]
np.log(gf0) #[204, 16, 230, 34]

# class singleton
galois.GF(3, 5) is galois.GF(3, 5) #True
galois.GF(3, 5, repr='int') is galois.GF(3, 5, repr='poly') #True
# isomorphic, but has different arithmetic
galois.GF(3, 5) is galois.GF(3, 5, irreducible_poly="x^5 + x^2 + x + 2") #False


GF3x5.ufunc_modes #jit-lookup jit-calculate
GF2x100 = galois.GF(2, 100)
GF2x100.ufunc_modes #python-calculate


# a*s+b*t=gcd
galois.egcd(a=5, b=7) #(gcd=1,s=3,t=-2)

list(galois.primitive_roots(11)) #2,6,7,8
galois.primitive_root(11) #2


## prime field
GF7 = galois.GF(7)
print(GF7.properties) #str
GF7.name #GF(7)
GF7.characteristic #7
GF7.degree #1
GF7.order #7
GF7.irreducible_poly #Poly(x + 4, GF(7))
GF7.is_prime_field #True
GF7.is_primitive_poly #True
GF7.primitive_element #GF7(3)
GF7.primitive_elements #3,5
GF7.element_repr #int
GF7.elements
print(GF7.arithmetic_table('+'))
print(GF7.arithmetic_table('-'))
print(GF7.arithmetic_table('*'))
print(GF7.arithmetic_table('/'))
print(GF7.repr_table()) #default to primitive=GF7(3)
print(GF7.repr_table(GF7(5)))
GF7(2).multiplicative_order() #3, so not a primitive element
gf0 = GF7(3)
gf1 = GF7(5)
gf1**(-1) #3
np.reciprocal(gf1) #3
gf0 / gf1 #2


## extension field
GF3x2 = galois.GF(3, 2)
print(GF3x2.properties) #str
GF3x2.name #GF(3^2)
GF3x2.characteristic #3
GF3x2.degree #2
GF3x2.order #9
GF3x2.irreducible_poly #Poly(x^2 + 2x + 2, GF(3))
GF3x2.is_prime_field #False
GF3x2.is_primitive_poly #True
GF3x2.primitive_element #GF3x2(3)
GF3x2.element_repr #int
GF3x2.primitive_elements #3,5,6,7
with GF3x2.repr('poly'):
    # print(GF3x2.arithmetic_table('/'))
    print(GF3x2.repr_table(GF7(3)))

## GF(3)[x]
GF3 = galois.GF(3)
gfp0 = galois.Poly([1,2], field=GF3) #x+2
gfp1 = galois.Poly([1,1], field=GF3) #x+1
