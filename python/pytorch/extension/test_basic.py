import numpy as np
import torch

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


class MyLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x_input, weight, bias=None):
        ctx.save_for_backward(x_input, weight)
        ret = torch.matmul(x_input, weight.t())
        if bias is not None:
            ret += bias
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        x_input, weight = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.matmul(grad_output, weight)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(grad_output.t(), x_input)
        if (len(ctx.needs_input_grad)>2) and (ctx.needs_input_grad[2]):
            grad_bias = grad_output.sum(0)
        return grad_x, grad_weight, grad_bias


def test_MyLinear(N0=23, N1=3, N2=5):
    # see https://pytorch.org/docs/stable/notes/extending.html#extending-autograd
    torch0 = torch.randn(N0, N1, requires_grad=True)
    torch1 = torch.randn(N0, N2)
    fc0 = torch.nn.Linear(N1, N2)
    torch.sum(fc0(torch0)*torch1).backward()
    ret_ = [x.grad.numpy().copy() for x in (torch0,fc0.weight,fc0.bias)]
    torch0.grad.zero_()
    fc0.zero_grad()
    torch.sum(MyLinear.apply(torch0, fc0.weight, fc0.bias)*torch1).backward()
    ret0 = [x.grad.numpy().copy() for x in (torch0,fc0.weight,fc0.bias)]
    assert all(hfe(x,y)<1e-4 for x,y in zip(ret_,ret0))

    torch0.grad.zero_()
    fc0 = torch.nn.Linear(N1, N2, bias=False)
    torch.sum(fc0(torch0)*torch1).backward()
    ret_ = [x.grad.numpy().copy() for x in (torch0,fc0.weight)]
    torch0.grad.zero_()
    fc0.zero_grad()
    torch.sum(MyLinear.apply(torch0, fc0.weight)*torch1).backward()
    ret0 = [x.grad.numpy().copy() for x in (torch0,fc0.weight)]
    assert all(hfe(x,y)<1e-4 for x,y in zip(ret_,ret0))


class MyReLU(torch.autograd.Function):

    @staticmethod
    def forward(context, x_input):
        context.save_for_backward(x_input)
        return x_input.clamp(min=0)

    @staticmethod
    def backward(context, grad_output):
        x_input, = context.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x_input < 0] = 0
        return grad_input

def test_MyReLU_gradient():
    torch0 = torch.tensor(np0, requires_grad=True)
    torch0 = torch.randn(2, 3, dtype=torch.float32, requires_grad=True)
    torch1 = torch.randn(2, 3, dtype=torch.float32)
    torch.sum((torch.nn.functional.relu(torch0)**3) * torch1).backward()
    ret_ = torch0.grad.numpy().copy()

    torch0.grad.zero_()
    torch.sum((MyReLU.apply(torch0)**3) * torch1).backward()
    ret0 = torch0.grad.numpy().copy()
    assert hfe(ret_, torch0.grad.numpy()) < 1e-5

# TODO torch.autograd.gradcheck
# TODO torch.autograd.gradgradcheck
# TODO torch.allclose
# TODO profile
