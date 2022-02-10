syms x y integer %real positive
assumptions
assume([x y],'clear')
assume(x >= 0)
assumeAlso(x,'integer')

syms s(t) f(x,y) %declare without definition
diff(f,x,y)

sym('A' , [2,3]); %matrix symbol

phi = (1 + sqrt(sym(5)))/2;
simplify(f)
expand(f)
factor(g)

syms a b c
A = [a b c; c a b; b c a];
isAlways(sum(A(1,:)) == sum(A(2,:)));

sym('A%d%d', [2 4]);

sym(hilb(3));

symvar(f)

subs(a,x,x+1)%替换之后依旧是符号对象

b = sym2poly(f);
polyval(b,1)

sym(t, 'f');
a = 3602879701896397/36028797018963968;

digits(7)
sym(t,'d')

int(f,x,y)
int(int(f,x),y) %different from the above
int(taylor(F, x, 'ExpansionPoint', 0, 'Order', 10), x)

F = int(cos(x)/sqrt(1 + x^2), x, 0, 10);
vpa(F,5)

syms x y
solve(6*x^2 - 6*x^2*y + x*y^2 - x*y + y^3 - y^2 == 0, y)
vpasolve(h, x)

% ezplot(x^3 - 6*x^2 + 11*x - 6); %TODO replace with fplot
% ezplot((x^2 + y^2)^4 == (x^2 - y^2)^2, [-1 1]);
% ezplot3(t^2*sin(10*t), t^2*cos(10*t), t);

limit(f, x, inf)

invhilb(20)
