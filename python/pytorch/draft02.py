import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torchvision
import torch.nn.functional as F


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
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


tmp0 = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root=os.path.join('~','pytorch_data'), train=True, download=True, transform=tmp0)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

tmp0 = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root=os.path.join('~','pytorch_data'), train=False, download=True, transform=tmp0)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


device = torch.device('cuda:0') #cpu

net = LeNet5().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
# optimizer = torch.optim.Adam(net.parameters())
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50,150], gamma=0.1)

metric_history = defaultdict(list)
for ind_epoch in range(10):
    with tqdm(total=len(trainloader), desc='epoch-{}'.format(ind_epoch)) as pbar:
        net.train()
        train_correct = 0
        train_total = 0
        for ind_batch, (data_i, label_i) in enumerate(trainloader):
            data_i, label_i = data_i.to(device), label_i.to(device)
            optimizer.zero_grad()
            predict = net(data_i)
            loss = F.cross_entropy(predict, label_i)
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
