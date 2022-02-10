import os
import numpy as np
import torch


hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())


class MyDummyModel(torch.nn.Module):
    def __init__(self, N0, N1):
        super().__init__()
        self.fc1 = torch.nn.Linear(N0, N1)
    def forward(self, x):
        x = self.fc1(x)
        return x

# https://pytorch.org/tutorials/beginner/saving_loading_models.html
def test_torch_net_optimizer_serialization(N0=13, N1=3, N2=5):
    np0 = np.random.rand(N0, N1).astype(np.float32)
    torch0 = torch.tensor(np0)
    hf_loss = lambda x: torch.sum(x)
    net0 = MyDummyModel(N1, N2)
    optimizer0 = torch.optim.Adam(net0.parameters())

    # net0
    hf_loss(net0(torch0)).backward()
    optimizer0.step()
    optimizer0.zero_grad()
    torch.save(net0.state_dict(), hf_file('net0.pth'))
    torch.save(optimizer0.state_dict(), hf_file('optimizer0.pth'))

    hf_loss(net0(torch0)).backward()
    optimizer0.step()
    optimizer0.zero_grad()
    ret_ = hf_loss(net0(torch0)).item()

    # net1
    net1 = MyDummyModel(N1, N2)
    optimizer1 = torch.optim.Adam(net1.parameters())
    net1.load_state_dict(torch.load(hf_file('net0.pth')))
    optimizer1.load_state_dict(torch.load(hf_file('optimizer0.pth')))
    hf_loss(net1(torch0)).backward()
    optimizer1.step()
    optimizer1.zero_grad()
    ret0 = hf_loss(net1(torch0)).item()

    assert hfe(ret_, ret0) < 1e-4
