# link: https://en.wikibooks.org/wiki/Introducing_Julia/Arrays_and_tuples

## range
2:5
1:10:100 #start:step:stop
5:-1:2
range(2, stop=5)
range(2, stop=5, step=2)
range(2, length=4)
step(range(2, length=4, stop=5))
length(range(2, stop=5, step=2))

# contstruct
zeros(2, 3) #Matrix(Float64,2,3)
[1 2] #Matrix(Int64,1,2)
[1, 2] #Vector(Int64,2)
[2, 23, 0.233] #Vector(Float64,3)
["2", "23", "233"] #Vector(String,3)
[sin, cos, tan] #Vector(Function,3)
[1, "2", 3.0, sin, pi] #Vector(Any,5)
Int32[2, 23, 233] #Vector(Int64, 3)
Int64[]
String[]
[1 2 3; 4 5 6] #Matrix(Int64,2,3)
[[1, 2, 3] [4, 5, 6]] #Matrix(Int64,3,2)
[1:3 4:6] #Matrix(Int64,3,2)
[1:3; 4:6] #Vector(Int64,6)
[[1, 2, 3]; [4, 5, 6]] #Vector(Int64,6)
[[1, 2, 3], [4, 5, 6]] #Vector(Vector(Int64))
[1:3, 1:3] #Vector(UnitRange(Int64),2)
[1;; 2;; 3;; 4] #Matrix(Int64,1,4)
[1; 2; 3; 4] #Vector(Int64,4)
[1 2 3 4] #Matrix(Int64,4,1)
[[1 1]; 2 3; [4 4]] #Matrix(Int64,3,2)
# [1, 1; 2, 3; 4, 4] #Wrong
Array(1:3) #Vector(Int64,3)

[1; 2;; 3; 4;; 5; 6;;; 7; 8;; 9; 10;; 11; 12] #Array(Int64,2,3,2)

sum(1/x^2 for x=1:1000)
[x^2 for x=1:3]
[(x,y) for x=1:3, y=1:4] #Matrix(Tuple(Int64,Int64),3,4)
[(x,y) for x=rand(Float64,1,2), y=rand(Float64,3,4)] #Array(Tuple(Float64,Float64),1,2,3,4)
[(x,y) for x=rand(Float64,1,2) for y=rand(Float64,3,4)] #Vector(Tuple(Float64,Float64),24)
[x^2 for x=1:10 if x!=5]
(x^2 for x=1:3) #generator
collect(x^2 for x in 1:3)

[1:3...] #splat operator
collect(1:3) #most time unnecessary
zeros(2, 3)
zeros(Int32, 2, 3)
zeros(Int32, (2,3))
rand(2, 3)
rand(Float32, 2, 3)
rand(Float32, (2,3))
rand([1,2,3], 2, 3)
rand(1:5, 2, 3)
randn(2, 3)
randn(Float32, 2, 3)
randn(Float32, (2,3))
fill(233, 2, 3)
fill(233, (2,3))
fill(Int32(233), (2,3))
fill("233", 2, 3)
z0 = Array{Int32}(undef, 5)
fill!(z0, 233)
reshape(1:12, 3, 4) #bad type
reshape(1:12, (3, 4)) #bad type
reshape(collect(1:12), (3,4))
# trues(2, 3) #ones(Bool, (2,3))
# ones() trues() falses() fill() fill!()
# do NOT use Vector() Matrix()

Array{Int64}(undef, 5) #Array{type}(xx, dims...), undef means uninitialized
Array{Int64}(undef, 2, 3, 3)

Array{Int}([])
Array{Int}[]
Int[]

[[1, 2], [3,4]] #Vector(Vector(Int64,2),2)
Array([[1, 2], [3,4]]) #Vector(Vector(Int64,2),2)
Array[[1, 2], [3,4]] #Vector(Array,2)

[1:3, 4:6] #Array{UnitRange{Int64},1}
Array([1:3, 4:6]) #Array{UnitRange{Int64},1}
Array[1:3, 4:6] #Array{Array,1}
z0 = Array[1:3, 4:6]
push!(z0, [233])

## concatenation
cat([1,2], [3,4], dims=1) #[1,2,3,4]
cat([1,2], [3,4], dims=2) #[1 3; 2 4]
cat([1 2], [3 4], dims=1) #[1 2; 3 4]
cat([1 2], [3 4], dims=2) #[1 2 3 4]


## indexing
a = collect(0:10:100)
a[1], a[2], a[end], a[end-1], a[2:2:end], a[[3,6,2]]
3 in a #in(3, a)
a[[true, true, false, true, true, true, false, true, false, false, false]]
a2 = [1 2 3; 4 5 6; 7 8 9]
a2[2], a2[1,2], a2[1,3], a2[:,2], a2[2,:], a2[2:3,:], a2[:]
# getindex
# setindex!

a = collect(1:5)
a.*2, a./2, a.*a, a./a
a==a, a.==a
reshape(a, 1, :) * reshape(a, :, 1)
reshape(a, 1, :) .* reshape(a, :, 1)
a = rand(0:10, 10, 10)
a[a.==0] .= 233
collect(1:100)[rand(1:end)]

# find indexing
a1 = [2,23,233,233,23,2]
a2 = [2 23 233; 233 23 2]
smallprimes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
23 in a2
findfirst(isequal(23), a1)
findfirst(x->(x==23), a2)
findall(in(a1), a2) #TODO why not use datatype-set
findall(iseven, a1) #isinteger
findnext(isequal(23), a1, 3)
filter(isodd, a1) #filter!
CartesianIndices(a2)[5]
LinearIndices(a2)[1,3]
# findmax() extrema() findmax() maximum()

# misc function
a1 = [2,23,233]
a2 = [2 23 233; 233 23 2]
ndims(a2)
size(a2), size(a2, 1), size(a2, 2)
length(a2)
count(!iszero, a2)
any(isodd, a1)
all(isodd, a1)

# 矢量化+大量临时存储空间分配 vs forloop+编译优化

# set operation: union() intersect() in() setdiff()

# using Statistics
# using Combinatorics

a1 = [23]
push!(a1, 233)
pushfirst!(a1, 2)
# splice!() pop!() popfirst!() deleteat!()


## named tuple
a = (x=2, y=23, z=233)
a.x, a.y, a.z, a[:x], a[:y], a[:z]
fieldnames(typeof(a))
