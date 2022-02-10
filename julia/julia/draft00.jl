# julia draft00.jl 2 23 233
println("hello world")
println(PROGRAM_FILE)
for x in ARGS;
    println(x);
end

# type
typemin(Int64)
typemax(Int64)
for x in [Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128]
    println("$(x): [$(typemin(x)), $(typemax(x))]")
end
typemin(Int64) == typemax(Int64)+1

# bool
true
false

# int
233 #Int64
23333333333333333333 #Int128
0b10 #UInt8 binary
0o010 #UInt8 octal
0x233 #UInt8 hexadecimal
10_000
bitstring(233)

# float
0.5 #Float64
0.5f0 #Float32
0.000_05
bitstring(2.33)
Inf16
Inf32
Inf64 #Inf
NaN16 #NaN32 NaN64 NaN
# isequal() isfinite() isinf() isnan()

pi
π
sqrt(100)

# type conversion

# operator
2 ^ 3
(^)(2, 3)
[2,3,3].^3
(^).([2,3,3], 3)
# == != < > <= ≤ >= ≥
# chained comparisons
# short-circuit evaluation
# operator precedence
# division functions.... (complicated): ÷ div()
÷# \div\tab

# complex number
1 + 2im #complex(1, 2)
# real() imag() conj() abs() abs2() angle()

# rational number
2//3
# numerator() denominator()


# character
'x', '\t', '\u2200' #∀
Int('x'), Char(120)
Char(0x110000), isvalid(Char, 0x110000) #invalid character, not checked in Char()
'A'<'a', 'c'-'a', 'A'+1


# string
"hello world", "hello \"world\""
x = "hello world"; x[1], x[end]
length("\u2200 x"), lastindex("\u2200 x") #lastindex() is NOT in general the same as length()
SubString("hello world") #typeof()=SubString
# chop() chomp() strip()
x = "\u2200 x \u2203 y" #x[2] x[3] fail, nextind(x, 1), prevind(x,4,1), prevind(x,lastindex(x),2), x[prevind(x,end,2)]
for x in "\u2200 x \u2203 y"
    println(x)
end
collect(eachindex("\u2200 x \u2203 y")) #codeunit() codeunits() ncodeunits()
string("hello", " ", "world"), "hello" * " " * "world"
x="hello"; y="world"; "$x $y"
"1+2=$(1+2)", "\$"
string(1), string(true)
"""233"""
"a"<"b", "c"=="c", "d"!="e"
findfirst(isequal('3'), "233"), findlast(isequal('3'), "233") #findnext() findprev() occursin()
join(["2","23","233"], "-")
repeat("233-", 3)

## regex string
r"^\s*(?:#|$)" #typeof() is Regex
# occursin()

## byte array literals
b"\xff"

## versino number literals
v"0.2"

## raw string literals
raw"\\"


# function
hf0(x) = x^2 + 2*x - 1
function hf1(x)
    return x^2 + 2*x - 1 #keyword "return" is optional
end
x -> x^2 + 2*x - 1
# varargs arguments
# optional arguments
# keyword arguments
hf2(x) = begin
 return 233*x
end
# do ... end
(sqrt∘+)(3, 6) #\circ\tab composition
1:10 |> sum |> - #piping

1+2+3, +(1,2,3)
# [A B C] hcat()
# [A; B; C] vcat()
# [A B; C D] hvcat()
# A' adjoint()
# A[i] getindex()
# A[i]=x setindex!()
# A.n getproperty()
# A.n=x setproperty!()

# tuple
(1, "hello", round)
(x=2, y=23, z=233).x

x = 1; y = 2;
if x < y
    println("x < y")
elseif x > y
    println("x > y")
else
    println("x == y")
end

true ? 1 : 0 #error no space

function fact(n::Int)
    n>=0 || error("n must be non-negative")
    n==0 ? 1 : (n*fact(n-1))
end


# dict
Dict([("A",233),("B",2333)])
Dict("A"=>233, "B"=>2333)
