import os
import collections
import numpy as np
from tqdm import tqdm
import torch
import torchvision

TORCH_DATA = os.path.join('~','pytorch_data')


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 20)
        self.fc2 = torch.nn.Linear(20, 1)
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x).view(-1)
        return x


np_data = np.random.randn(2333, 10)
np_label = np.random.randn(2333)
torch_data = torch.tensor(np_data, dtype=torch.float32)
torch_label = torch.tensor(np_label, dtype=torch.float32)

net = Net()
# net = torch.nn.Sequential(torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 1))
net.parameters()
net.fc1.weight
net.fc1.bias

ret_ = net(torch_data)
weight = {
    'fc1/kernel': net.fc1.weight.detach().numpy(),
    'fc1/bias': net.fc1.bias.detach().numpy(),
    'fc2/kernel': net.fc2.weight.detach().numpy(),
    'fc2/bias': net.fc2.bias.detach().numpy(),
}
x = np_data @ weight['fc1/kernel'].T + weight['fc1/bias']
x = np.maximum(x, 0) #relu
ret0 = (x @ weight['fc2/kernel'].T + weight['fc2/bias'])[:,0]
assert np.abs(ret_.detach().numpy()-ret0).max() < 1e-4


# save and reload
ret_ = net(torch_data).detach().numpy()
torch.save(net.state_dict(), 'tbd00.pth')
net00 = Net()
net00.load_state_dict(torch.load('tbd00.pth'))
ret0 = net00(torch_data).detach().numpy()
assert np.abs(ret_-ret0).max() < 1e-5


# train step by hand
learning_rate = 0.01
hf_loss = torch.nn.MSELoss()
loss = hf_loss(net(torch_data), torch_label)
net.zero_grad()
loss.backward()
for x in net.parameters():
    # x(torch.nn.parameter.Parameter) requires_grad=True
    # x.data(torch.tensor) requires_grad=False
    # x.grad(torch.tensor) requires_grad=False
    x.data.sub_(x.grad * learning_rate)


trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch_data, torch_label), batch_size=4, shuffle=True)
testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch_data, torch_label), batch_size=8)

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
device = torch.device('cuda:0') #torch.device('cpu')
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


class LeNet5(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16*5*5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)
        self.num_classes = 10

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def demo_LeNet5():
    tmp0 = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=TORCH_DATA, train=True, download=True, transform=tmp0)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    tmp0 = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root=TORCH_DATA, train=False, download=True, transform=tmp0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


    device = torch.device('cuda:0') #cpu

    net = LeNet5().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    # optimizer = torch.optim.Adam(net.parameters())
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50,150], gamma=0.1)

    metric_history = collections.defaultdict(list)
    for ind_epoch in range(10):
        with tqdm(total=len(trainloader), desc='epoch-{}'.format(ind_epoch)) as pbar:
            net.train()
            train_correct = 0
            train_total = 0
            for ind_batch, (data_i, label_i) in enumerate(trainloader):
                data_i, label_i = data_i.to(device), label_i.to(device)
                optimizer.zero_grad()
                predict = net(data_i)
                loss = torch.nn.functional.cross_entropy(predict, label_i)
                loss.backward()
                optimizer.step()

                train_correct += (predict.max(dim=1)[1]==label_i).sum().item()
                train_total += label_i.shape[0]
                metric_history['train-loss'].append(loss.item())
                if ind_batch+1 < len(trainloader):
                    pbar.set_postfix({'loss':'{:5.3}'.format(loss.item()), 'acc':'{:4.3}%'.format(100*train_correct/train_total)})
                    pbar.update() #move the last update to val
            metric_history['train-acc'].append(train_correct / train_total)
            lr_scheduler.step()

            net.eval()
            with torch.no_grad():
                tmp0 = ((net(x.to(device)), y.to(device)) for x,y in testloader)
                val_acc = sum((x.max(dim=1)[1]==y).sum().item() for x,y in tmp0) / len(testloader.dataset)
            metric_history['val-acc'].append(val_acc)
            pbar.set_postfix({'acc':'{:4.3}%'.format(100*train_correct/train_total), 'val-acc':'{:4.3}%'.format(val_acc*100)})
            pbar.update()
