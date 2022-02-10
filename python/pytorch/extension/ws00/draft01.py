import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension

my_pytorch_ext = torch.utils.cpp_extension.load(name="my_pytorch_ext", sources=["lltm.cpp"])

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

class PyLLTM(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super().__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = torch.nn.Parameter(
            torch.empty(3 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        old_h, old_cell = state
        X = torch.cat([old_h, input], dim=1)
        gate_weights = F.linear(X, self.weights, self.bias)
        gates = gate_weights.chunk(3, dim=1)
        input_gate = torch.sigmoid(gates[0])
        output_gate = torch.sigmoid(gates[1])
        candidate_cell = F.elu(gates[2])
        new_cell = old_cell + candidate_cell * input_gate
        new_h = torch.tanh(new_cell) * output_gate
        return new_h, new_cell


class LLTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        outputs = my_pytorch_ext.forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        outputs = my_pytorch_ext.backward(
            grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors)
        d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class CppLLTM(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super().__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = torch.nn.Parameter(
            torch.empty(3 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return LLTMFunction.apply(input, self.weights, self.bias, *state)


device = 'cuda' #cuda cpu

batch_size = 2
input_features = 32
state_size = 128
X = torch.randn(batch_size, input_features, device=device)
h = torch.randn(batch_size, state_size, device=device)
C = torch.randn(batch_size, state_size, device=device)
weights = torch.randn(3*state_size, input_features+state_size, device=device)
bias = torch.randn(3*state_size, device=device)

rnn = PyLLTM(input_features, state_size).to(device=device)
rnn.weights.data.copy_(weights)
rnn.bias.data.copy_(bias)
with torch.no_grad():
    ret_h_, ret_C_ = rnn(X, (h, C))

rnn = CppLLTM(input_features, state_size).to(device=device)
rnn.weights.data.copy_(weights)
rnn.bias.data.copy_(bias)
with torch.no_grad():
    ret_h0, ret_C0 = rnn(X, (h,C))

assert hfe(ret_h_.cpu().numpy(), ret_h0.cpu().numpy()) < 1e-5
assert hfe(ret_C_.cpu().numpy(), ret_C0.cpu().numpy()) < 1e-5
