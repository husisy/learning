import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision

TORCH_DATA_ROOT = os.path.expanduser(os.path.join('~', 'torch_data'))

class MNISTDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        dim_list = [784, 256, 256, 1]
        self.fc_list = torch.nn.ModuleList([torch.nn.Linear(x, y) for x,y in zip(dim_list,dim_list[1:])])

    def forward(self, x):
        x = x.reshape(-1, 784)
        for ind0 in range(len(self.fc_list)-1):
            x = self.fc_list[ind0](x)
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
        logit = self.fc_list[-1](x).reshape(-1)
        prob = torch.sigmoid(logit)
        return logit,prob

class MNISTGenerator(torch.nn.Module):
    def __init__(self, dim_latent):
        super().__init__()
        dim_list = [dim_latent, 256, 256, 784]
        self.fc_list = torch.nn.ModuleList([torch.nn.Linear(x, y) for x,y in zip(dim_list,dim_list[1:])])

    def forward(self, x):
        for ind0 in range(len(self.fc_list)-1):
            x = self.fc_list[ind0](x)
            x = torch.nn.functional.relu(x)
        x = torch.tanh(self.fc_list[-1](x)) #put in range [-1,1]
        x = x.reshape(-1, 1, 28, 28)
        return x


class MNISTGAN(torch.nn.Module):
    def __init__(self, dim_latent, use_wgan=False, wgan_clipping=0.005):
        super().__init__()
        self.dim_latent = dim_latent
        self.use_wgan = use_wgan
        self.wgan_clipping = wgan_clipping
        self.gen = MNISTGenerator(dim_latent)
        self.disc = MNISTDiscriminator()

    def forward_disc(self, real_image, x_latent):
        logit_real,prob_real = self.disc(real_image)
        image_gen = self.gen(x_latent)
        logit_gen,prob_gen = self.disc(image_gen)
        if self.use_wgan:
            loss = -torch.mean(logit_real) + torch.mean(logit_gen)
        else:
            one = torch.ones_like(logit_real)
            zero = torch.zeros_like(logit_real)
            loss = F.binary_cross_entropy_with_logits(logit_real, one).mean() + F.binary_cross_entropy_with_logits(logit_gen, zero).mean()
        return loss, prob_real, prob_gen

    def forward_gen(self, x_latent):
        image_gen = self.gen(x_latent)
        logit_gen,prob_gen = self.disc(image_gen)
        one = torch.ones_like(logit_gen)
        if self.use_wgan:
            loss = -torch.mean(logit_gen)
        else:
            loss = F.binary_cross_entropy_with_logits(logit_gen, one).mean()
        return loss, prob_gen

    def clip_disc_weight(self):
        for fc_i in self.disc.fc_list:
            fc_i.weight.data.clip_(-self.wgan_clipping, self.wgan_clipping)

    def generate_image(self, x_latent=None):
        self.eval()
        device = self.gen.fc_list[0].weight.device
        if x_latent is None:
            tmp0 = torch.randn(1, self.dim_latent).to(device)
        elif not hasattr(x_latent, '__len__'):
            x_latent = int(x_latent)
            tmp0 = torch.randn(x_latent, self.dim_latent).to(device)
        else:
            tmp0 = x_latent if isinstance(x_latent,torch.Tensor) else torch.tensor(x_latent)
        with torch.no_grad():
            ret = (self.gen(tmp0).cpu().detach().numpy()>0).astype(np.float32)[:,0]
        if x_latent is None:
            ret = ret[0]
        return ret


num_disc_step = 5
dim_latent = 100
batch_size = 128
num_epoch = 100
learning_rate = 0.0003
use_wgan = True
device = 'cuda'

tmp0 = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])
ds_train = torchvision.datasets.MNIST(root=TORCH_DATA_ROOT, train=True, transform=tmp0, download=True)
dl_train = torch.utils.data.DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True)


model = MNISTGAN(dim_latent, use_wgan).to(device)
optimizer_gen = torch.optim.Adam(model.gen.parameters(), lr=learning_rate)
optimizer_disc = torch.optim.Adam(model.disc.parameters(), lr=learning_rate)
history_metric = {'loss_g':[], 'loss_d':[]}
model.train()
for ind_epoch in range(num_epoch):
    with tqdm(dl_train, total=len(dl_train), desc=f'[epoch={ind_epoch}]') as pbar:
        for ind_step,(image_real,_) in enumerate(pbar):
            image_real = image_real.to(device)
            for _ in range(num_disc_step):
                x_latent = torch.randn(len(image_real), dim_latent).to(device)

                optimizer_disc.zero_grad()
                loss,_,_ = model.forward_disc(image_real, x_latent)
                loss.backward()
                optimizer_disc.step()
                history_metric['loss_d'].append(loss.item())
                if use_wgan:
                    model.clip_disc_weight()

            optimizer_gen.zero_grad()
            loss,_ = model.forward_gen(x_latent)
            loss.backward()
            optimizer_gen.step()
            history_metric['loss_g'].append(loss.item())

            if ind_step%10==0:
                tmp0 = {k:f'{v[-1]:.5f}' for k,v in history_metric.items()}
                pbar.set_postfix(**tmp0)


image_gen = model.generate_image(100)
z0 = np.pad(image_gen.reshape(10,10,28,28), [(0,0),(0,0),(2,2),(2,2)], mode='constant', constant_values=1).transpose(0,2,1,3).reshape(320,320)
fig,ax = plt.subplots()
ax.axis('off')
fig.tight_layout()
