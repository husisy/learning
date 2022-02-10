import numpy as np
import torch

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


def test_torch_hessian(N0=3):
    np0 = np.random.rand(N0).astype(np.float32)
    # np1 = np.prod(np0)**2
    hessian_np = 4 * np.prod(np0)**2 / (np0[:,np.newaxis]*np0)
    hessian_np[np.arange(N0), np.arange(N0)] /= 2

    torch0 = torch.tensor(np0, requires_grad=True)
    torch1 = torch.prod(torch0)**2
    gradient = torch.autograd.grad(torch1, torch0, create_graph=True)[0]
    ret0 = []
    for ind0 in range(N0):
        gradient[ind0].backward(retain_graph=(ind0<(N0-1)))
        ret0.append(torch0.grad.detach().numpy().copy())
        torch0.grad.zero_()
    ret0 = np.stack(ret0)
    assert hfe(hessian_np, ret0) < 1e-5


class MyModel00(torch.nn.Module):
    def __init__(self):
        super(MyModel00, self).__init__()
        self.fc0 = torch.nn.Linear(5, 13)
        self.fc1 = torch.nn.Linear(13, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = torch.nn.functional.relu(x)
        x = self.fc1(x)[:,0]
        return x

def _gradient_accumulate_np_random_dataset(random_state=None):
    np_random_generator = np.random.RandomState(random_state)
    data = np_random_generator.randn(13, 5).astype(np.float32)
    label = np_random_generator.randn(13).astype(np.float32)
    return data, label

def test_gradient_accumulate():
    np_random_state_list = [5,7,11,3]
    net0 = MyModel00()
    grad_list0 = []
    for random_state_i in np_random_state_list:
        net0.zero_grad()
        data,label = _gradient_accumulate_np_random_dataset(random_state_i)
        data = torch.Tensor(data)
        predict = net0(data)
        loss = torch.mean((predict-torch.Tensor(label))**2)
        loss.backward()
        grad_list0.append([x.grad.numpy().copy() for x in net0.parameters()])

    grad_list1 = []
    net0.zero_grad()
    for random_state_i in np_random_state_list:
        data,label = _gradient_accumulate_np_random_dataset(random_state_i)
        data = torch.Tensor(data)
        predict = net0(data)
        loss = torch.mean((predict-torch.Tensor(label))**2)
        loss.backward()
        grad_list1.append([x.grad.numpy().copy() for x in net0.parameters()])

    for grad_i,grad_accumulate in zip(zip(*grad_list0), zip(*grad_list1)):
        tmp0 = [sum(grad_i[:(x+1)]) for x in range(len(grad_i))]
        assert all(hfe(x,y)<1e-4 for x,y in zip(tmp0,grad_accumulate))
