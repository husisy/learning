import numpy as np
import torch

assert torch.cuda.is_available()

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
to_np = lambda x,dtype=np.float32: x.cpu().detach().numpy().astype(dtype)
hfe16 = lambda x,y,eps=1e-3: hfe(x.astype(np.float32), y.astype(np.float32), eps)

def test_float16_matmul():
    np0 = np.random.rand(3,1024).astype(np.float16) + 1
    np1 = np.random.rand(1024,3).astype(np.float16) + 1
    np2 = (np0.astype(np.float32) @ np1.astype(np.float32)).astype(np.float16)
    np3 = np0 @ np1 #decimals between 1024 and 2048 is 2, see https://en.wikipedia.org/wiki/Half-precision_floating-point_format
    assert hfe16(np2, np3) < 1e-3
    device = torch.device('cuda')
    torch0 = torch.tensor(np0, requires_grad=True, device=device)
    torch1 = torch.tensor(np1, requires_grad=True, device=device)
    torch2 = torch.matmul(torch0, torch1)
    assert hfe16(np3, to_np(torch2)) < 1e-3


def test_cuda_amp_GradScaler(shape=(2,3), amp_gradscale=0.233):
    device = torch.device('cuda')
    np0 = np.random.randn(*shape).astype(np.float16)
    np1 = np.random.randn(*shape).astype(np.float16)

    torch0 = torch.tensor(np0, requires_grad=True, device=device)
    torch1 = torch.tensor(np1, device=device)
    scaler = torch.cuda.amp.GradScaler(init_scale=amp_gradscale)
    loss0 = (torch0*torch1).sum()
    loss1 = scaler.scale(loss0)
    loss1.backward()
    torch0.grad #float16

    assert hfe(loss0.item()*amp_gradscale, loss1.item()) < 1e-3
    assert hfe16(np1*amp_gradscale, to_np(torch0.grad)) < 1e-3


class DummyNet(torch.nn.Module):
    def __init__(self, torch0):
        super(DummyNet, self).__init__()
        self.torch0 = torch.nn.Parameter(torch0)
        self.hack = [None]
    def forward(self, torch1):
        torch2 = (self.torch0*torch1).sum() + 1
        torch2.retain_grad()
        self.hack[0] = torch2
        loss0 = torch2 - 1
        return loss0


def test_amp_scale_retain_grad():
    shape = (2,3)
    amp_gradscale = 0.233
    device = torch.device('cuda')
    np0 = np.random.randn(*shape).astype(np.float32)
    np1 = np.random.randn(*shape).astype(np.float16)

    scaler = torch.cuda.amp.GradScaler(init_scale=amp_gradscale)
    net = DummyNet(torch.tensor(np0, requires_grad=True, device=device))
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    torch1 = torch.tensor(np1, device=device)
    with torch.cuda.amp.autocast():
        loss = net(torch1)
    scaler.scale(loss).backward()
    assert hfe16(np1*amp_gradscale, to_np(net.torch0.grad)) < 1e-3
    assert hfe16(to_np(net.hack[0].grad), np.array(amp_gradscale)) < 1e-3

    scaler.unscale_(optimizer)
    assert hfe16(np1, to_np(net.torch0.grad)) < 1e-3
    assert hfe16(to_np(net.hack[0].grad), np.array(amp_gradscale)) < 1e-3 #still unscaled
