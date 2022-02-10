import os
import timeit
import numpy as np

import tvm
# import tvm.testing
# from tvm import te

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())


def new_c(M, N, dtype, dev):
    ret = tvm.nd.array(np.zeros((M, N), dtype=dtype), dev)
    return ret

def build_op_matmul_simple(target, M, K, N):
    k = tvm.te.reduce_axis((0, K), "k")
    A = tvm.te.placeholder((M, K), name="A")
    B = tvm.te.placeholder((K, N), name="B")
    C = tvm.te.compute((M, N), lambda x, y: tvm.te.sum(A[x, k] * B[k, y], axis=k), name="C")
    s = tvm.te.create_schedule(C.op)
    op_matmul = tvm.build(s, [A, B, C], target=target, name="mmult")
    return op_matmul

def build_op_matmul_blocking(target, M, K, N, bn=32):
    # 32*32*4(float32) = 4 Kilo-Byte
    # on p720, bn=128-256 works best
    k = tvm.te.reduce_axis((0, K), "k")
    A = tvm.te.placeholder((M, K), name="A")
    B = tvm.te.placeholder((K, N), name="B")
    C = tvm.te.compute((M, N), lambda x, y: tvm.te.sum(A[x, k] * B[k, y], axis=k), name="C")
    s = tvm.te.create_schedule(C.op)
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=4)
    s[C].reorder(xo, yo, ko, ki, xi, yi)
    op_matmul = tvm.build(s, [A, B, C], target=target, name="mmult")
    return op_matmul

def build_op_matmul_vectorization(target, M, K, N, bn=32):
    k = tvm.te.reduce_axis((0, K), "k")
    A = tvm.te.placeholder((M, K), name="A")
    B = tvm.te.placeholder((K, N), name="B")
    C = tvm.te.compute((M, N), lambda x, y: tvm.te.sum(A[x, k] * B[k, y], axis=k), name="C")
    s = tvm.te.create_schedule(C.op)
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=4)
    s[C].reorder(xo, yo, ko, xi, ki, yi) #loop permutation
    s[C].vectorize(yi)
    op_matmul = tvm.build(s, [A, B, C], target=target, name="mmult")
    return op_matmul


def build_op_matmul_packing(target, M, K, N, bn=32):
    k = tvm.te.reduce_axis((0, K), "k")
    A = tvm.te.placeholder((M, K), name="A")
    B = tvm.te.placeholder((K, N), name="B")
    packedB = tvm.te.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name="packedB")
    C = tvm.te.compute(
        (M, N),
        lambda x, y: tvm.te.sum(A[x, k] * packedB[y // bn, k, tvm.tir.indexmod(y, bn)], axis=k),
        name="C",
    )
    s = tvm.te.create_schedule(C.op)
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=4)
    s[C].reorder(xo, yo, ko, xi, ki, yi) #loop permutation
    s[C].vectorize(yi)
    x, y, z = s[packedB].op.axis
    s[packedB].vectorize(z)
    s[packedB].parallel(x)
    op_matmul = tvm.build(s, [A, B, C], target=target, name="mmult")
    return op_matmul


def build_op_matmul_optimize5(target, M, K, N, bn=32):
    k = tvm.te.reduce_axis((0, K), "k")
    A = tvm.te.placeholder((M, K), name="A")
    B = tvm.te.placeholder((K, N), name="B")
    packedB = tvm.te.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name="packedB")
    C = tvm.te.compute(
        (M, N),
        lambda x, y: tvm.te.sum(A[x, k] * packedB[y // bn, k, tvm.tir.indexmod(y, bn)], axis=k),
        name="C",
    )
    s = tvm.te.create_schedule(C.op)
    CC = s.cache_write(C, "global")
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    s[CC].compute_at(s[C], yo)
    xc, yc = s[CC].op.axis
    ko, ki = s[CC].split(s[CC].op.reduce_axis[0], factor=4)
    s[CC].reorder(ko, xc, ki, yc)
    s[CC].unroll(ki)
    s[CC].vectorize(yc)
    x, y, z = s[packedB].op.axis
    s[packedB].vectorize(z)
    s[packedB].parallel(x)
    op_matmul = tvm.build(s, [A, B, C], target=target, name="mmult")
    return op_matmul


def build_op_matmul_optimize6(target, M, K, N, bn=32):
    k = tvm.te.reduce_axis((0, K), "k")
    A = tvm.te.placeholder((M, K), name="A")
    B = tvm.te.placeholder((K, N), name="B")
    packedB = tvm.te.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name="packedB")
    C = tvm.te.compute(
        (M, N),
        lambda x, y: tvm.te.sum(A[x, k] * packedB[y // bn, k, tvm.tir.indexmod(y, bn)], axis=k),
        name="C",
    )
    s = tvm.te.create_schedule(C.op)
    CC = s.cache_write(C, "global")
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    s[CC].compute_at(s[C], yo)
    xc, yc = s[CC].op.axis
    ko, ki = s[CC].split(s[CC].op.reduce_axis[0], factor=4)
    s[CC].reorder(ko, xc, ki, yc)
    s[CC].unroll(ki)
    s[CC].vectorize(yc)
    s[C].parallel(xo)
    x, y, z = s[packedB].op.axis
    s[packedB].vectorize(z)
    s[packedB].parallel(x)
    op_matmul = tvm.build(s, [A, B, C], target=target, name="mmult")
    return op_matmul



M = 2048
K = 2048
N = 2048
dtype = "float32"
num_time_repeat = 10

tmp0 = timeit.timeit(
    setup='import numpy as np\n'
    f'M,K,N,dtype = {M},{K},{N},"{dtype}"\n'
    'a = np.random.rand(M, K).astype(dtype)\n'
    'b = np.random.rand(K, N).astype(dtype)\n'
    'c = np.zeros((M,N), dtype=dtype)',
    stmt="answer = np.dot(a, b, out=c)",
    number=num_time_repeat,
)
np_running_time = tmp0 / num_time_repeat
print(f'numpy: {np_running_time}')

target = tvm.target.Target(target="llvm -mcpu=skylake-avx512", host="llvm -mcpu=skylake-avx512")
dev = tvm.device(target.kind.name, 0)

a = tvm.nd.array(np.random.rand(M, K).astype(dtype), dev)
b = tvm.nd.array(np.random.rand(K, N).astype(dtype), dev)
ret_ = np.dot(a.numpy(), b.numpy())
answer = np.dot(a.numpy(), b.numpy())


# op = build_op_matmul_simple(target, M, K, N)
# c = new_c(M, N, dtype, dev)
# op(a, b, c)
# assert hfe(c.numpy(), np.dot(a.numpy(), b.numpy())) < 1e-5
# c = new_c(M, N, dtype, dev)
# mean_time = op.time_evaluator(op.entry_name, dev, number=3)(a,b,c).mean #repeating 100 times is quite long for op_simple
# print(f'tvm-simlpe: {mean_time}')

my_bn = 64

op = build_op_matmul_blocking(target, M, K, N, bn=my_bn)
c = new_c(M, N, dtype, dev)
op(a, b, c)
assert hfe(c.numpy(), np.dot(a.numpy(), b.numpy())) < 1e-5
c = new_c(M, N, dtype, dev)
mean_time = op.time_evaluator(op.entry_name, dev, number=num_time_repeat)(a,b,c).mean
print(f'tvm-blocking: {mean_time}')


op = build_op_matmul_vectorization(target, M, K, N, bn=my_bn)
c = new_c(M, N, dtype, dev)
op(a, b, c)
assert hfe(c.numpy(), np.dot(a.numpy(), b.numpy())) < 1e-5
c = new_c(M, N, dtype, dev)
mean_time = op.time_evaluator(op.entry_name, dev, number=num_time_repeat)(a,b,c).mean
print(f'tvm-vectorization: {mean_time}')


op = build_op_matmul_packing(target, M, K, N, bn=my_bn)
c = new_c(M, N, dtype, dev)
op(a, b, c)
assert hfe(c.numpy(), np.dot(a.numpy(), b.numpy())) < 1e-5
c = new_c(M, N, dtype, dev)
mean_time = op.time_evaluator(op.entry_name, dev, number=num_time_repeat)(a,b,c).mean
print(f'tvm-packing: {mean_time}')


op = build_op_matmul_optimize5(target, M, K, N, bn=my_bn)
c = new_c(M, N, dtype, dev)
op(a, b, c)
assert hfe(c.numpy(), np.dot(a.numpy(), b.numpy())) < 1e-5
c = new_c(M, N, dtype, dev)
mean_time = op.time_evaluator(op.entry_name, dev, number=num_time_repeat)(a,b,c).mean
print(f'tvm-optimize5: {mean_time}')


op = build_op_matmul_optimize6(target, M, K, N, bn=my_bn)
c = new_c(M, N, dtype, dev)
op(a, b, c)
assert hfe(c.numpy(), np.dot(a.numpy(), b.numpy())) < 1e-5
c = new_c(M, N, dtype, dev)
mean_time = op.time_evaluator(op.entry_name, dev, number=num_time_repeat)(a,b,c).mean
print(f'tvm-optimize6: {mean_time}')
