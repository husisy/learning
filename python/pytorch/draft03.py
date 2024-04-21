import numpy as np
import torch
import torch.distributed.pipeline.sync

class DummyModel00(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 5)

    def forward(self, x):
        x = self.fc(x)
        return x

model_fp32 = DummyModel00()
# create a quantized model instance
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights
input_fp32 = torch.randn(3, 4)
res = model_int8(input_fp32)


fc1 = torch.nn.Linear(7, 5).cuda(0)
fc2 = torch.nn.Linear(5, 3).cuda(1)
model = torch.nn.Sequential(fc1, fc2)
model = torch.distributed.pipeline.sync.Pipe(model, chunks=8)
torch0 = torch.rand(16, 7).cuda(0)
output_rref = model(torch0) #ERROR Current RPC agent is not set!


## jit
class MyCell(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        ret = torch.tanh(self.linear(x) + h)
        return ret, ret

my_cell = MyCell()
x, h = torch.rand(3, 4), torch.rand(3, 4)
traced_cell = torch.jit.trace(my_cell, (x, h))
traced_cell.graph
print(traced_cell.code)
traced_cell(x, h)


## torch.jit.script for control-flow
class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

class MyCell(torch.nn.Module):
    def __init__(self, dg):
        super(MyCell, self).__init__()
        self.dg = dg
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h, new_h

scripted_gate = torch.jit.script(MyDecisionGate()) #control flow
my_cell = MyCell(scripted_gate)
scripted_cell = torch.jit.script(my_cell)
print(scripted_gate.code)
print(scripted_cell.code)

x, h = torch.rand(3, 4), torch.rand(3, 4)
tmp0 = MyCell(MyDecisionGate())
tmp0.linear.weight.data[:] = scripted_cell.linear.weight.data
tmp0.linear.bias.data[:] = scripted_cell.linear.bias.data
assert torch.abs(tmp0(x, h)[0] - scripted_cell(x, h)[0]).max().item() < 1e-6



## Mixing Scripting and Tracing
class MyRNNLoop(torch.nn.Module):
    def __init__(self, x, h):
        super().__init__()
        self.cell = torch.jit.trace(MyCell(scripted_gate), (x, h))

    def forward(self, xs):
        h, y = torch.zeros(3, 4), torch.zeros(3, 4)
        for i in range(xs.size(0)):
            y, h = self.cell(xs[i], h)
        return y, h

class WrapRNN(torch.nn.Module):
    def __init__(self, x, h):
        super().__init__()
        self.loop = torch.jit.script(MyRNNLoop(x, h))

    def forward(self, xs):
        y, h = self.loop(xs)
        return torch.relu(y)

rnn_loop = torch.jit.script(MyRNNLoop(x, h))
xs = torch.rand(10, 3, 4)
traced = torch.jit.trace(WrapRNN(x, h), xs)
z0 = traced(xs)
traced.save('tbd00.pt')
z1 = torch.jit.load('tbd00.pt')
assert torch.abs(z0 - z1(xs)).max().item() < 1e-6


## torch.func
x0 = torch.randn([])
x1 = torch.func.grad(torch.sin)(x0)
assert torch.abs(x1 - torch.cos(x0)).item() < 1e-6
x2 = torch.func.grad(torch.func.grad(torch.sin))(x0)
assert torch.abs(x2 + torch.sin(x0)).item() < 1e-6



batch_size = 3
N1 = 5
x0 = torch.randn(7, N1)
x1 = torch.randn(batch_size, N1, requires_grad=True)
def hf0(x):
    assert x.ndim == 1
    ret = torch.nn.functional.relu(x0 @ x)
    return ret
x2 = torch.stack([hf0(x) for x in x1])
x3 = torch.func.vmap(hf0)(x1)
assert torch.abs(x2 - x3).max().item() < 1e-6


def hf0(weights, data, label):
    assert data.ndim == 1
    y = data.dot(weights).relu()
    loss = ((y - label) ** 2).mean()
    return loss
batch_size, feature_size = 3, 5
weights = torch.randn(feature_size, requires_grad=True)
data = torch.randn(batch_size, feature_size)
label = torch.randn(batch_size)
hf1 = torch.func.vmap(torch.func.grad(hf0), in_dims=(None,0,0))
grad = hf1(weights, data, label) #grad.shape=(3,5)


x0 = torch.randn(5)
x1 = torch.randn(3, 5)
hf0 = lambda x: x1 @ torch.sin(x)
cotangent = torch.randn(3)
outputs, vjp_fn = torch.func.vjp(hf0, x0)
vjps = vjp_fn(cotangent) #backward mode


x0 = torch.randn(5)
x1 = torch.randn(3, 5)
hf0 = lambda x: x1 @ torch.sin(x)
dual_tangent = torch.randn(5)
output, out_tangent = torch.func.jvp(hf0, (x0,), (dual_tangent,)) #forward mode


x0 = torch.randn(5)
ret_ = torch.diag(torch.cos(x0))
ret_reverse_mode = torch.func.jacrev(torch.sin)(x0)
assert torch.allclose(ret_reverse_mode, ret_)
ret_forward_mode = torch.func.jacfwd(torch.sin)(x0)
assert torch.allclose(ret_forward_mode, ret_)


x0 = torch.randn(64, 5)
jacobian = torch.func.vmap(torch.func.jacrev(torch.sin))(x0)
assert jacobian.shape == (64, 5, 5)



hf0 = lambda x: x.sin().sum()
x0 = torch.randn(5)
hess = torch.func.hessian(hf0)(x0)
hessian0 = torch.func.jacrev(torch.func.jacrev(hf0))(x0)
hessian1 = torch.func.jacfwd(torch.func.jacrev(hf0))(x0)
