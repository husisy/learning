# https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html
import os
import time
import tempfile
import itertools
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

hf_data = lambda *x: os.path.join('..','data', *x)
# https://download.pytorch.org/tutorial/hymenoptera_data.zip

tmp0 = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.RandomCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
ds_train = torchvision.datasets.ImageFolder(hf_data('hymenoptera_data','train'), transform=tmp0)
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=16, shuffle=True, num_workers=8)
tmp0 = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
ds_val = torchvision.datasets.ImageFolder(hf_data('hymenoptera_data','val'), transform=tmp0)
dl_val = torch.utils.data.DataLoader(ds_val, batch_size=16, shuffle=False, num_workers=8)


def plt_show_image(ds0):
    ind0 = np.random.randint(len(ds0), size=16)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_list = [np.clip(ds0[x][0].numpy().transpose(1,2,0)*std+mean, 0, 1) for x in ind0]
    label_list = [ds0.classes[ds0[x][1]] for x in ind0]
    fig,tmp0 = plt.subplots(4,4,figsize=(5,5))
    ax_list = [y for x in tmp0 for y in x]
    for ax,image,label in zip(ax_list, image_list, label_list):
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label)
    fig.tight_layout()
    return fig
# fig = plt_show_image(ds_train)
# fig.savefig('tbd00.png')


# use quantized resnet18 as FIXED feature extractor
model_resnet18_quantized = torchvision.models.quantization.resnet18(pretrained=True, progress=True, quantize=True)
model = torch.nn.Sequential(
    model_resnet18_quantized.quant,  # Quantize the input
    model_resnet18_quantized.conv1,
    model_resnet18_quantized.bn1,
    model_resnet18_quantized.relu,
    model_resnet18_quantized.maxpool,
    model_resnet18_quantized.layer1,
    model_resnet18_quantized.layer2,
    model_resnet18_quantized.layer3,
    model_resnet18_quantized.layer4,
    model_resnet18_quantized.avgpool,
    model_resnet18_quantized.dequant,  # Dequantize the output
    torch.nn.Flatten(1),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(model_resnet18_quantized.fc.in_features, 2),
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# quantized model can only run on CPU
train_history = {'train_loss':[], 'train_acc':[], 'val_acc':[]}
for epoch in range(25):
    model.train()
    with tqdm(dl_train, description=f'[epoch={epoch}]') as pbar:
        for data_i, label_i in pbar:
            optimizer.zero_grad()
            logits = model(data_i)
            prediction = torch.argmax(logits, dim=1)
            loss = F.cross_entropy(logits, label_i)
            loss.backward()
            optimizer.step()
            train_history['train_loss'].append(loss.item())
            train_history['train_acc'].append((prediction==label_i).sum()/len(data_i))
            tmp0 = train_history['train_loss'][-5:]
            tmp1 = train_history['train_acc'][-5:]
            pbar.set_postfix(loss=f'{sum(tmp0)/len(tmp0):.3f}', acc=f'{sum(tmp1)/len(tmp1):.3f}')
    lr_scheduler.step()
    model.eval()
    with torch.no_grad():
        num_correct = sum([(torch.argmax(model(x), dim=1)==y).sum().item() for x,y in dl_val])
        train_history['val_acc'].append(num_correct / len(ds_val))
    print('val-acc:', train_history['val_acc'][-1])



## Quantization Aware Training
model_resnet18 = torchvision.models.quantization.resnet18(pretrained=True, progress=True, quantize=False)
model_resnet18.train()
model_resnet18.fuse_model()
model = torch.nn.Sequential(
    torch.nn.Sequential(
        model_resnet18.quant,  # Quantize the input
        model_resnet18.conv1,
        model_resnet18.bn1,
        model_resnet18.relu,
        model_resnet18.maxpool,
        model_resnet18.layer1,
        model_resnet18.layer2,
        model_resnet18.layer3,
        model_resnet18.layer4,
        model_resnet18.avgpool,
        model_resnet18.dequant,  # Dequantize the output
    ),
    torch.nn.Flatten(1),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(model_resnet18.fc.in_features, 2),
)
device = 'cuda'
model[0].qconfig = torch.quantization.default_qat_qconfig  # Use default QAT configuration
model = torch.quantization.prepare_qat(model, inplace=True).to(device) #QAT can be trained on GPU

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.1)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
train_history = {'train_loss':[], 'train_acc':[], 'val_acc':[]}
for epoch in range(25):
    model.train()
    with tqdm(dl_train, desc=f'[epoch={epoch}]') as pbar:
        for data_i, label_i in pbar:
            data_i, label_i = data_i.to(device), label_i.to(device)
            optimizer.zero_grad()
            logits = model(data_i)
            prediction = torch.argmax(logits, dim=1)
            loss = F.cross_entropy(logits, label_i)
            loss.backward()
            optimizer.step()
            train_history['train_loss'].append(loss.item())
            train_history['train_acc'].append((prediction==label_i).sum().item()/len(data_i))
            tmp0 = train_history['train_loss'][-5:]
            tmp1 = train_history['train_acc'][-5:]
            pbar.set_postfix(loss=f'{sum(tmp0)/len(tmp0):.3f}', acc=f'{sum(tmp1)/len(tmp1):.3f}')
    lr_scheduler.step()
    model.eval()
    with torch.no_grad():
        num_correct = sum([(torch.argmax(model(x.to(device)), dim=1)==y.to(device)).sum().item() for x,y in dl_val])
        train_history['val_acc'].append(num_correct / len(ds_val))
    print('val-acc:', train_history['val_acc'][-1])
model = model.cpu() #inplace operation
model_quantized = torch.quantization.convert(model, inplace=False)

model_quantized.eval()
with torch.no_grad():
    num_correct = sum([(torch.argmax(model_quantized(x), dim=1)==y).sum().item() for x,y in dl_val])
    acc_quantized = num_correct / len(ds_val)
print('acc(model,test):', train_history['val_acc'][-1])
print('acc(model_quantized,test):', acc_quantized)
