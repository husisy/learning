import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from collections import defaultdict

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 20)
        self.fc2 = torch.nn.Linear(20, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x).view(-1)
        return x


np_data = np.random.randn(2333, 10)
np_label = np.random.randn(2333)
torch_data = torch.tensor(np_data, dtype=torch.float32)
torch_label = torch.tensor(np_label, dtype=torch.float32)

net = Net()
hf_loss = torch.nn.MSELoss()
# net = torch.nn.Sequential(
#     torch.nn.Linear(10, 20),
#     torch.nn.ReLU(),
#     torch.nn.Linear(20, 1),
# )


net.parameters()
net.fc1.weight
net.fc1.bias


# predict, and compare by hand
ret_ = net(torch_data)
weight = {
    'fc1/kernel': net.fc1.weight.detach().numpy(),
    'fc1/bias': net.fc1.bias.detach().numpy(),
    'fc2/kernel': net.fc2.weight.detach().numpy(),
    'fc2/bias': net.fc2.bias.detach().numpy(),
}
x = np_data
x = x @ weight['fc1/kernel'].T + weight['fc1/bias']
x = np.maximum(x, 0) #relu
ret0 = (x @ weight['fc2/kernel'].T + weight['fc2/bias'])[:,0]
assert hfe(ret_.detach().numpy(), ret0) < 1e-4


# save and reload
ret_ = net(torch_data).detach().numpy()
torch.save(net.state_dict(), 'tbd00.pth')
net00 = Net()
net00.load_state_dict(torch.load('tbd00.pth'))
ret0 = net00(torch_data).detach().numpy()
assert hfe(ret_, ret0) < 1e-5


# train step by hand
learning_rate = 0.01
loss = hf_loss(net(torch_data), torch_label)
net.zero_grad()
loss.backward()
for x in net.parameters():
    # x(torch.nn.parameter.Parameter) requires_grad=True
    # x.data(torch.tensor) requires_grad=False
    # x.grad(torch.tensor) requires_grad=False
    x.data.sub_(x.grad * learning_rate)


# Dataloader
trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch_data, torch_label), batch_size=4, shuffle=True)
testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch_data, torch_label), batch_size=8)


# optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_history = []
for epoch in range(2):
    net.train()
    with tqdm(trainloader) as pbar:
        for data_i,label_i in pbar:
            optimizer.zero_grad()
            loss = hf_loss(net(data_i), label_i)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            pbar.set_postfix(train_loss='{:5.3}'.format(sum(loss_history[-10:])/10))
with torch.no_grad():
    net.eval()
    mse_val = sum(((net(x)-y)**2).sum() for x,y in testloader).item() / len(testloader.dataset)


# train on GPU
assert torch.cuda.is_available()
device_gpu = torch.device('cuda:0')
device_cpu = torch.device('cpu')
device = device_gpu
net = Net().to(device)

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_history = []
for epoch in range(2):
    net.train()
    with tqdm(trainloader) as pbar:
        for data_i,label_i in pbar:
            optimizer.zero_grad()
            loss = hf_loss(net(data_i.to(device)), label_i.to(device))
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            pbar.set_postfix(train_loss='{:5.3}'.format(sum(loss_history[-10:])/10))
with torch.no_grad():
    net.eval()
    tmp0 = ((net(x.to(device))-y.to(device))**2 for x,y in testloader)
    mse_val = sum(x.sum() for x in tmp0).item() / len(testloader.dataset)
