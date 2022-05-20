using Test
using CUDA
using BenchmarkTools


function sequential_add!(y, x)
    for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

function parallel_add!(y, x)
    Threads.@threads for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

function gpu_add!(y, x)
    CUDA.@sync y .+= x
    return nothing
end

N = 2^20;
x = zeros(Float32, N);
y = zeros(Float32, N);
fill!(x, 1);
fill!(y, 2);
sequential_add!(y, x)
@test all(y .== 3.0f0)
@btime sequential_add!($y, $x) #71.518 μs (0 allocations: 0 bytes)

fill!(y, 2);
parallel_add!(y, x);
@test all(y .== 3.0f0)
@btime parallel_add!($y, $x) #76.477 μs (6 allocations: 544 bytes)

x_d = CUDA.zeros(Float32, N);
y_d = CUDA.zeros(Float32, N);
CUDA.fill!(x_d, 1.0f0);
CUDA.fill!(y_d, 2.0f0);
y_d .+= x_d
@test all(Array(y_d) .== 3.0f0)
@btime gpu_add!($y_d, $x_d) # 47.353 μs (7 allocations: 480 bytes)
