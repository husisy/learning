import numpy as np
import torch
import torch.nn.functional as F

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


# dtype
torch.get_default_dtype()
torch.int8, torch.int16, torch.int32, torch.int64 #torch.long
torch.float16, torch.float32, torch.float64
torch.complex32, torch.complex64, torch.complex128 #almost not support


# create
torch.empty(3, 5, dtype=torch.int32)
torch.zeros(3, 5)
torch.tensor([2,23,233]) #int64
torch.tensor([0.2, 0.23, 0.233]) #float32
torch.rand(3, 5)
torch.randn(3, 5)


# torch.Tensor
x = torch.rand(3, 5)
x.dtype
x.size()
x.shape
x.view(5, -1)
torch.Size([3, 5]) #? for what


# + operation
x0 = torch.rand(3,5)
x1 = torch.rand(3,5)
x0 + x1
torch.add(x0, x1)
x2 = torch.empty(*x0.size(), dtype=x0.dtype)
torch.add(x0, x1, out=x2)
x3 = torch.zeros(6, 5, dtype=x0.dtype)
torch.add(x0, x1, out=x3[::2])
x4 = x0.clone().detach()
x4.add_(x1) #in-place
x0.add(x1) #out-place


# misc00
torch.rand(1).item() #for single element shape=(1,) or shape=()
torch.rand(1)[0].item()


# numpy bridge
torch0 = torch.rand(3,5)
np0 = torch0.numpy()
np.shares_memory(np0, torch0.numpy()) #True
np1 = np.random.rand(3,5)
torch1 = torch.from_numpy(np1)
np.shares_memory(np1, torch1.numpy()) #True


# cuda tensor
assert torch.cuda.is_available()
device = torch.device('cuda')
torch0_gpu = torch.rand(3, 5, device=device)
torch1 = torch.rand(3,5)
torch1_gpu = torch1.to(device)
torch.cuda.device_count()


# requires_grad
torch0 = torch.rand(3, 5, requires_grad=False) #default
torch0.requires_grad #False
torch0.data.requires_grad #False
torch1 = torch0**2
torch1.requires_grad #False
torch1.data.requires_grad #False

torch0 = torch.rand(3, 5, requires_grad=True)
torch0.requires_grad #True
torch0.data.requires_grad #False
torch1 = torch0**2
torch1.requires_grad #True
torch1.data.requires_grad #False
torch2 = torch0.data**2
torch2.requires_grad #False
with torch.no_grad():
    torch3 = torch0**2
    print(torch3.requires_grad) #False


torch0 = torch.rand(3, 5)
torch0.requires_grad_(True) #change the flag in-place


# autograd
torch0 = torch.rand(3, 5, requires_grad=True)
torch1 = torch.sum(torch.sin(torch0))
torch1.grad_fn
torch1.grad_fn.next_functions
torch1.grad_fn.next_functions[0][0].next_functions
torch1.backward()
torch0.grad


# autograd self-assign, NEVER should pass
# torch.autograd.set_detect_anomaly(True)
# torch0 = torch.randn(3, 5, requires_grad=True)
# torch0[0,0] = torch0[0,0]**2
# torch.sum(torch0).backward()


# TensorDataset, DataLoader
np0 = np.random.randn(233, 3)
np1 = np.random.randn(233)
dataset0 = torch.utils.data.TensorDataset(torch.tensor(np0), torch.tensor(np1))
len(dataset0)
dataset0[1:3]
dataloader0 = torch.utils.data.DataLoader(dataset0, batch_size=4)
len(dataloader0)
next(iter(dataloader0))
list(dataloader0)


# misc
# torch.set_num_threads(1)
# print(torch.__config__.parallel_info())
