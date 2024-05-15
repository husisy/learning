import numpy as np
import torch


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
