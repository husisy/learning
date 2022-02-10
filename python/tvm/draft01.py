import os
import timeit
import numpy as np

import tvm
import tvm.contrib.utils
import tvm.contrib.cc


hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())

def build_op_myadd(target, parallel=False, simd_factor=None):
    n = tvm.te.var("n")
    A = tvm.te.placeholder((n,), name="A")
    B = tvm.te.placeholder((n,), name="B")
    C = tvm.te.compute(A.shape, lambda i: A[i] + B[i], name="C")
    s = tvm.te.create_schedule(C.op)
    if parallel:
        if simd_factor is None:
            s[C].parallel(C.op.axis[0])
        else:
            # simd_factor should be chosen to match the number of threads appropriate for
            # your CPU. This will vary depending on architecture, but a good rule is
            # setting this factor to equal the number of available CPU cores.
            outer,inner = s[C].split(C.op.axis[0], factor=simd_factor)
            print(f'outer={outer}, inner={inner}')
            s[C].parallel(outer)
            s[C].vectorize(inner)
    op_add = tvm.build(s, [A, B, C], target, name='myadd')
    return op_add,s,A,B,C

def build_op_myadd_gpu(target):
    n = tvm.te.var("n")
    A = tvm.te.placeholder((n,), name="A")
    B = tvm.te.placeholder((n,), name="B")
    C = tvm.te.compute(A.shape, lambda i: A[i] + B[i], name="C")
    s = tvm.te.create_schedule(C.op)
    bx, tx = s[C].split(C.op.axis[0], factor=64)
    # bind the iteration axis bx and tx to threads in the GPU compute grid
    s[C].bind(bx, tvm.te.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.te.thread_axis("threadIdx.x"))
    op_add = tvm.build(s, [A, B, C], target=target, name="myadd")
    return op_add,s,A,B,C

def tvm_detect_gpu_target(name):
    tmp0 = ['cuda','rocm', 'opencl']
    tmp1 = [None] + [x for x in tmp0 if name.startswith(x)]
    ret = tmp1[-1]
    return ret

def tvm_save_compiled_module(op, target_name, directory, op_name='myop'):
    print(f'[WARNING/tvm_save_compiled_module()] use op.export_library() instead')
    hf0 = lambda *x: os.path.join(directory, *x)
    op.save(hf0(f'{op_name}.o'))
    target_name = tvm_detect_gpu_target(target_name)
    if target_name is not None:
        tmp0 = {'cuda':'.ptx', 'rocm':'.hsaco', 'opencl':'.cl'}[target_name]
        op.imported_modules[0].save(hf0(f'{op_name}{tmp0}'))

def tvm_load_compiled_module(op_name, target_name, directory):
    print(f'[WARNING/tvm_load_compiled_module()] use tvm.runtime.load_module() instead')
    hf0 = lambda *x: os.path.join(directory, *x)
    op = tvm.runtime.load_module(hf0(f'{op_name}.o'))
    target_name = tvm_detect_gpu_target(target_name)
    if target_name is not None:
        tmp0 = {'cuda':'.ptx', 'rocm':'.hsaco', 'opencl':'.cl'}[target_name]
        op.import_module(tvm.runtime.load_module(hf0(f'{op_name}{tmp0}')))
    return op

target = tvm.target.Target(target='llvm -mcpu=skylake-avx512', host='llvm -mcpu=skylake-avx512')

# fadd,s,A,B,C = build_op_myadd(target, parallel=False)
# fadd,s,A,B,C = build_op_myadd(target, parallel=True)
fadd,s,A,B,C = build_op_myadd(target, parallel=True, simd_factor=4)
# print(tvm.lower(s, [A,B,C], simple_mode=True)) #Intermediate Representation (IR)
# print(fadd.get_source())

# tvm_save_compiled_module(fadd, target.kind.name, hf_file(), op_name='myadd')
# fadd1 = tvm_load_compiled_module('myadd', target.kind.name, hf_file())
fadd.export_library(hf_file('myadd1.so'))
fadd1 = tvm.runtime.load_module(hf_file('myadd1.so'))

dev = tvm.device(target.kind.name, 0)

n = 32768
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
fadd(a, b, c)
assert hfe(a.numpy()+b.numpy(), c.numpy()) < 1e-5
fadd1(a, b, c)
assert hfe(a.numpy()+b.numpy(), c.numpy()) < 1e-5

num_repeat = 100
tvm_running_time = fadd.time_evaluator(fadd.entry_name, dev, number=num_repeat)(a, b, c).mean
print(f'tvm: {tvm_running_time}')

tmp0 = timeit.timeit(
    setup="import numpy as np\n"
    f"n = {n}\n"
    'dtype = np.float32\n'
    "a = np.random.rand(n, 1).astype(dtype)\n"
    "b = np.random.rand(n, 1).astype(dtype)\n"
    "c = np.zeros_like(b)\n",
    stmt="answer = a + b",
    number=num_repeat,
)
np_running_time = tmp0 / num_repeat
print(f'numpy: {np_running_time}')


# cuda (NVIDIA GPUs), rocm (Radeon GPUS), OpenCL (opencl).
target_gpu = tvm.target.Target(target="cuda", host="llvm -mcpu=skylake-avx512")
fadd,s,A,B,C = build_op_myadd_gpu(target_gpu)

fadd.export_library(hf_file('myadd2.so')) #not save new if already exists, so name should be different
fadd1 = tvm.runtime.load_module(hf_file('myadd2.so'))


dev = tvm.device(target_gpu.kind.name, 0)

n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
fadd(a, b, c)
assert hfe(c.numpy(), a.numpy() + b.numpy()) < 1e-5
fadd1(a, b, c)
assert hfe(c.numpy(), a.numpy() + b.numpy()) < 1e-5
# print(fadd.imported_modules[0].get_source())
