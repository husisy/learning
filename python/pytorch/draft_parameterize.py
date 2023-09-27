# https://pytorch.org/tutorials/intermediate/parametrizations.html
import torch

class Symmetric(torch.nn.Module):
    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)


class Skew(torch.nn.Module):
    def forward(self, X):
        A = X.triu(1)
        return A - A.transpose(-1, -2)

layer = torch.nn.Linear(3, 3)
z0 = layer.weight #removed
torch.nn.utils.parametrize.register_parametrization(layer, "weight", Symmetric())
list(layer.named_parameters())
z1 = layer.parametrizations.weight.original
z2 = layer.weight #not share memory with z0

x0 = layer.weight + layer.weight**2 #computed twice
with torch.nn.utils.parametrize.cached():
    x1 = layer.weight + layer.weight**2 #just compute once


class DummyParametrize(torch.nn.Module):
    def forward(self, x, y):
        return x + y
    def right_inverse(self, X):
        tmp0 = torch.rand_like(X)
        ret = X - tmp0, tmp0
        return ret

N0 = 3
N1 = 5
layer = torch.nn.Linear(N0, N1)
torch.nn.utils.parametrize.register_parametrization(layer, "weight", DummyParametrize())


layer = torch.nn.Linear(3, 3)
print("Before:")
print(layer)
print(layer.weight)
torch.nn.utils.parametrize.register_parametrization(layer, "weight", Skew())
print("\nParametrized:")
print(layer)
print(layer.weight)
torch.nn.utils.parametrize.remove_parametrizations(layer, "weight")
# torch.nn.utils.parametrize.remove_parametrizations(layer, "weight", leave_parametrized=False)
print("\nAfter")
print(layer)
print(layer.weight)
