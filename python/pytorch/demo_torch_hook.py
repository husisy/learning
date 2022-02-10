import numpy as np
import torch
import torch.nn.functional as F

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


def demo_module_register_hook():
    z0 = []
    def hf_hook0(module, input, output):
        z0.append(module) #fc0
        z0.append(input) #(tuple,torch.tensor,1)
        z0.append(output) #torch.tensor
    def hf_hook1(module, grad_input, grad_output):
        z0.append(module) #fc0
        z0.append(grad_input) #(tuple,torch.tensor/None,3)
        z0.append(grad_output) #(tuple,torch.tensor,1)
    N0 = 5
    N1 = 2
    N2 = 3
    torch0 = torch.randn(N0, N1, requires_grad=True)
    fc0 = torch.nn.Linear(N1, N2)
    _ = fc0(torch0) #initialize weights
    fc0.register_forward_hook(hf_hook0)
    fc0.register_backward_hook(hf_hook1)
    fc0(torch0).sum().backward()
    print(z0)


def demo_tensor_hook():
    z0 = []
    def hf_hook(grad):
        z0.append(grad) #torch.tensor
    torch0 = torch.randn(3, 5, requires_grad=True)
    torch1 = torch.randn(*torch0.shape)
    torch2 = torch0**2
    torch2.register_hook(hf_hook)
    torch2.backward(torch1)
    assert hfe(z0[0].numpy(), torch1.numpy()) < 1e-4
