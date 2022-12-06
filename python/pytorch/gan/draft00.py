import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision

DATA_DIR = os.path.expanduser(os.path.join('~','torch_data'))

class MNISTGenerator(torch.nn.Module):
    def __init__(self, dim_latent):
        super().__init__()
        dim_list = [dim_latent, 256, 512, 1024, 784]
        self.fc_list = torch.nn.ParameterList([torch.nn.Linear(x, y) for x,y in zip(dim_list,dim_list[1:])])
        self.bn_list = torch.nn.ParameterList([torch.nn.BatchNorm1d(x,momentum=0.5) for x in dim_list[1:-1]])

    def forward(self, x):
        for ind0 in range(len(self.bn_list)):
            x = self.fc_list[ind0](x)
            x = self.bn_list[ind0](x)
            x = torch.nn.functional.leaky_relu(x)
        x = self.fc_list[-1](x).reshape(-1, 28, 28)
        return x

class MNISTDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        dim_list = [784, 512, 256, 1]
        self.fc_list = torch.nn.ParameterList([torch.nn.Linear(x, y) for x,y in zip(dim_list,dim_list[1:])])
        self.bn_list = torch.nn.ParameterList([torch.nn.BatchNorm1d(x,momentum=0.2) for x in dim_list[1:-1]])

    def forward(self, image):
        x = image.reshape(-1, 784)
        for ind0 in range(len(self.bn_list)):
            # print(ind0, x.shape, self.fc_list[ind0])
            x = self.fc_list[ind0](x)
            x = self.bn_list[ind0](x)
            x = torch.nn.functional.leaky_relu(x)
        logit = self.fc_list[-1](x).reshape(-1) #logit>0 means predicting it as a real image, otherwise fake image
        prob_real = torch.sigmoid(logit) #prob_real>0.5 means predicting it as a real image, otherwise fake image
        return prob_real


class MNISTGAN(torch.nn.Module):
    def __init__(self, dim_latent):
        super().__init__()
        self.gen = MNISTGenerator(dim_latent)
        self.disc = MNISTDiscriminator()

    def forward_disc(self, real_image, x_latent):
        prob_real = self.disc(real_image)
        image_gen = self.gen(x_latent)
        prob_gen = self.disc(image_gen)
        loss = torch.mean(prob_gen) - torch.mean(prob_real)
        return loss, prob_real, prob_gen

    def forward_gen(self, x_latent):
        image_gen = self.gen(x_latent)
        prob_gen = self.disc(image_gen)
        loss = -torch.mean(prob_gen)
        return loss, prob_gen


def load_mnist_dataset(batch_size):
    tmp0 = torchvision.transforms.ToTensor()
    ds_train = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=tmp0)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    tmp0 = torchvision.transforms.ToTensor()
    ds_test = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=tmp0)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    return ds_train,dl_train,ds_test,dl_test



dim_latent = 100
batch_size = 64
num_epoch = 10

ds_mnist,dl_mnist,_,_ = load_mnist_dataset(batch_size)

model = MNISTGAN(dim_latent)
optimizer_gen = torch.optim.Adam(model.gen.parameters(), lr=0.002)
optimizer_disc = torch.optim.Adam(model.disc.parameters(), lr=0.0001)
history_metric = {'loss_g':[], 'loss_d':[]}
model.train()
for ind_epoch in range(num_epoch): #1 minute
    with tqdm(dl_mnist, total=len(dl_mnist), desc=f'[epoch={ind_epoch}]') as pbar:
        for image_real,_ in pbar:
            x_latent = torch.randn(batch_size, dim_latent)

            optimizer_disc.zero_grad()
            loss,_,_ = model.forward_disc(image_real, x_latent)
            loss.backward()
            optimizer_disc.step()
            history_metric['loss_d'].append(loss.item())

            optimizer_gen.zero_grad()
            loss,_ = model.forward_gen(x_latent)
            loss.backward()
            optimizer_gen.step()
            history_metric['loss_g'].append(loss.item())

            tmp0 = {k:f'{v[-1]:.5f}' for k,v in history_metric.items()}
            pbar.set_postfix(**tmp0)

model.eval()
x_latent = torch.randn(25, dim_latent)
image_gen = model.gen(x_latent).detach().numpy()
fig,tmp0 = plt.subplots(5,5)
ax_list = [y for x in tmp0 for y in x]
for ind0 in range(25):
    ax_list[ind0].imshow(image_gen[ind0])
    ax_list[ind0].axis('off')
fig.tight_layout()
