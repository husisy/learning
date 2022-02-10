# link https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
import os
import torch
import numpy as np
import torchvision
from tqdm import tqdm
from collections import defaultdict

hf_data = lambda *x: os.path.join('data', *x)
assert os.path.exists(hf_data('hymenoptera_data')) #https://download.pytorch.org/tutorial/hymenoptera_data.zip

device = torch.device('cuda:0') #torch.device('cpu')

def show_dataset():
    import matplotlib.pyplot as plt
    plt.ion()
    tmp0,tmp1 = next(iter(trainloader))
    tmp0 = torchvision.utils.make_grid(tmp0).numpy().transpose(1,2,0)
    tmp0 = np.clip(tmp0 * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
    fig,ax = plt.subplots()
    ax.imshow(tmp0)
    ax.set_title(','.join(trainset.classes[x] for x in tmp1.numpy().tolist()))


tmp0 = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
trainset = torchvision.datasets.ImageFolder(hf_data('hymenoptera_data','train'), transform=tmp0)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)
len(trainset)
trainset.classes
trainset.class_to_idx

tmp0 = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
testset = torchvision.datasets.ImageFolder(hf_data('hymenoptera_data','val'), transform=tmp0)
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=4)


net = torchvision.models.resnet18(pretrained=True)
for x in net.parameters(): #fixed feature extractor
    x.requires_grad = False #TODO BN need more operation
tmp0 = list(set(list(net.modules())) - {net.fc})
tmp1 = (
    torch.nn.modules.batchnorm.BatchNorm1d,
    torch.nn.modules.batchnorm.BatchNorm2d,
    torch.nn.modules.batchnorm.BatchNorm3d,
)
bn_module = [x for x in tmp0 if isinstance(x, tmp1)] #net.train will change the state "bn.training"
for x in bn_module:
    x.eval()
net.fc = torch.nn.Linear(net.fc.in_features, 2)
net = net.to(device)
hf_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD([x for x in net.parameters() if x.requires_grad], lr=0.001, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 7, gamma=0.1)

# best_model_weight = copy.deepcopy(net.state_dict())
# net.load_state_dict(best_model_weight)


metric_history = defaultdict(list)
for ind_epoch in range(25):
    with tqdm(total=len(trainloader), desc='epoch-{}'.format(ind_epoch)) as pbar:
        net.train()
        for x in bn_module:
            x.eval()
        correct = 0
        train_correct = 0
        train_total = 0
        for ind_batch, (data_i, label_i) in enumerate(trainloader):
            data_i, label_i = data_i.to(device), label_i.to(device)
            optimizer.zero_grad()
            predict = net(data_i)
            loss = hf_loss(predict, label_i)
            loss.backward()
            optimizer.step()

            train_correct += (predict.max(dim=1)[1]==label_i).sum().item()
            train_total += label_i.size()[0]
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
