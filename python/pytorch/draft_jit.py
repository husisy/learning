import numpy as np
import torch


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
