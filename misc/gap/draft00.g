3-2;
7-9;
(9-7) * (5+6);
12/8; #3/2
2^10;
2^66; #arbitrary size integer

7 mod 3;

3<=5; #true
3<5; #true
3>5; #false
3>=5; #false

3=5; #false
3<>5; #true

not true; #false
true and false; #false
true or false; #true

#permutation
(1,2,3);
(1,2,3) * (1,2); #(2,3)
(1,2,3)^(-1); #(1,3,2)

# finite field
Z(8);

# complex root of unity
E(4);

# character and string
'a';
"aaa";

a := 10; #create variable
a = 10; #true
a := "233"; #re-assign value
NamesUserGVars(); #list all user-defined variables
last;
last2;
last3;

2+3;; #compress print-out


a := (1,2,3);; IsIdenticalObj(a, a); #true
b := (1,2,3);; IsIdenticalObj(a, b); #false
b := a;; IsIdenticalObj(a, b); #true

Factorial(5);
Gcd(6, 8);
Print(233, "\n");

hf0 := x -> x^3;
hf0(3);

# help
?Tutorial:Sets
?stabilizer
??stabilizer
?Reference:Read
