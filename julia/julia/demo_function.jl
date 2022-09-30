hf0(x::Float64, y::Float64) = 2x + y
hf0(2.0, 3.0)
# hf0(2.0, 3) #error
# hf0(Float32(2.0), 3.0) #error

hf1(x::Number, y::Number) = 2x - y
hf1(2.0, 3.0)
hf1(2.0, 3)
hf1(Float32(2.0), 3.0)


hf2(x::Int64) = "Int64::" * hf2(Int32(x))
hf2(x::Int32) = "Int32::" * hf2(Float64(x))
hf2(x::Float64) = "Float64::" * hf2(Float32(x))
hf2(x::Float32) = "Float32::" * string(x)
hf2(Int64(3)) #Int64::Int32::Float64::Float32::3.0
methods(hf2)


∘ #\circ<tab>

z0 = rand(4)
z1 = similar(z0)
@. z1 = sin(cos(z0))


[1:5;] |> (x->x.^2) |> sum |> inv #0.01818181818181818
