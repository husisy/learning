import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision

TORCH_DATA_ROOT = os.path.expanduser(os.path.join('~', 'torch_data'))

class MNISTDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        dim_list = [784, 256, 256, 11] #the last one is for fake label
        self.fc_list = torch.nn.ModuleList([torch.nn.Linear(x, y) for x,y in zip(dim_list,dim_list[1:])])

    def forward(self, x):
        x = x.reshape(-1, 784)
        for ind0 in range(len(self.fc_list)-1):
            x = self.fc_list[ind0](x)
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
        logit = self.fc_list[-1](x)
        prob = torch.nn.functional.softmax(logit, dim=-1)
        return logit,prob

class MNISTGenerator(torch.nn.Module):
    def __init__(self, dim_latent):
        super().__init__()
        dim_list = [dim_latent+10, 256, 256, 784]
        self.fc_list = torch.nn.ModuleList([torch.nn.Linear(x, y) for x,y in zip(dim_list,dim_list[1:])])

    def forward(self, x):
        for ind0 in range(len(self.fc_list)-1):
            x = self.fc_list[ind0](x)
            x = torch.nn.functional.relu(x)
        x = torch.tanh(self.fc_list[-1](x)) #put in range [-1,1]
        x = x.reshape(-1, 1, 28, 28)
        return x

class MNISTCGAN(torch.nn.Module):
    def __init__(self, dim_latent):
        super().__init__()
        self.dim_latent = dim_latent
        self.gen = MNISTGenerator(dim_latent)
        self.disc = MNISTDiscriminator()
        self.hf_loss = torch.nn.CrossEntropyLoss()

    def forward_disc(self, image_real, label_real, x_latent):
        logit_real,prob_real = self.disc(image_real)
        label_one_hot = torch.zeros(len(image_real), 10, dtype=image_real.dtype, device=image_real.device)
        label_one_hot[torch.arange(len(image_real)), label_real] = 1
        image_gen = self.gen(torch.concat([x_latent,label_one_hot], dim=1))
        logit_gen,prob_gen = self.disc(image_gen)
        loss = self.hf_loss(logit_real, label_real) + self.hf_loss(logit_gen, torch.ones_like(label_real)*10)
        # one = torch.ones_like(logit_real)
        # zero = torch.zeros_like(logit_real)
        # loss = F.binary_cross_entropy_with_logits(logit_real, one).mean() + F.binary_cross_entropy_with_logits(logit_gen, zero).mean()
        return loss, prob_real, prob_gen

    def forward_gen(self, x_latent, label):
        label_one_hot = torch.zeros(len(x_latent), 10, dtype=x_latent.dtype, device=x_latent.device)
        label_one_hot[torch.arange(len(x_latent)), label] = 1
        image_gen = self.gen(torch.concat([x_latent,label_one_hot], dim=1))
        logit_gen,prob_gen = self.disc(image_gen)
        loss = self.hf_loss(logit_gen, label)
        return loss, prob_gen

    def generate_image(self, label, x_latent=None):
        self.eval()
        device = self.gen.fc_list[0].weight.device
        if x_latent is None:
            tmp0 = torch.randn(1, self.dim_latent).to(device)
        elif not hasattr(x_latent, '__len__'):
            x_latent = int(x_latent)
            tmp0 = torch.randn(x_latent, self.dim_latent).to(device)
        else:
            tmp0 = x_latent if isinstance(x_latent,torch.Tensor) else torch.tensor(x_latent)
        tmp1 = torch.zeros(len(tmp0), 10, dtype=tmp0.dtype, device=tmp0.device)
        tmp1[:, label] = 1
        tmp0 = torch.concat([tmp0,tmp1], dim=1)
        with torch.no_grad():
            ret = (self.gen(tmp0).cpu().detach().numpy()>0).astype(np.float32)[:,0]
        if x_latent is None:
            ret = ret[0]
        return ret


num_disc_step = 5
dim_latent = 100
batch_size = 128
num_epoch = 100
device = 'cuda'

tmp0 = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])
ds_train = torchvision.datasets.MNIST(root=TORCH_DATA_ROOT, train=True, transform=tmp0, download=True)
dl_train = torch.utils.data.DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True)

model = MNISTCGAN(dim_latent).to(device)
optimizer_gen = torch.optim.Adam(model.gen.parameters(), lr=0.0003)
optimizer_disc = torch.optim.Adam(model.disc.parameters(), lr=0.0003)
history_metric = {'loss_g':[], 'loss_d':[]}
model.train()
for ind_epoch in range(num_epoch):
    with tqdm(dl_train, total=len(dl_train), desc=f'[epoch={ind_epoch}]') as pbar:
        for ind_step,(image_real,label_real) in enumerate(pbar):
            image_real = image_real.to(device)
            label_real = label_real.to(device)
            for _ in range(num_disc_step):
                x_latent = torch.randn(len(image_real), dim_latent).to(device)

                optimizer_disc.zero_grad()
                loss,_,_ = model.forward_disc(image_real, label_real, x_latent)
                loss.backward()
                optimizer_disc.step()
                history_metric['loss_d'].append(loss.item())

            optimizer_gen.zero_grad()
            loss,_ = model.forward_gen(x_latent, label_real)
            loss.backward()
            optimizer_gen.step()
            history_metric['loss_g'].append(loss.item())

            if ind_step%10==0:
                tmp0 = {k:f'{v[-1]:.5f}' for k,v in history_metric.items()}
                pbar.set_postfix(**tmp0)


image_gen = np.stack([model.generate_image(x, 10) for x in range(10)])
z0 = np.pad(image_gen, [(0,0),(0,0),(2,2),(2,2)], mode='constant', constant_values=1).transpose(0,2,1,3).reshape(320,320)
fig,ax = plt.subplots()
ax.imshow(z0)
ax.axis('off')
fig.tight_layout()
