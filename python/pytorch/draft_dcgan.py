import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision


def show_image(image_batch):
    import matplotlib.pyplot as plt
    plt.ion()
    tmp0 = torchvision.utils.make_grid(image_batch[:64], padding=2, normalize=True).numpy().transpose(1,2,0)
    fig,ax = plt.subplots()
    ax.axis('off')
    ax.imshow(tmp0)
    fig.tight_layout()


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.convt0 = torch.nn.ConvTranspose2d(100, 512, 4, stride=1, padding=0, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(512)
        self.convt1 = torch.nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(256)
        self.convt2 = torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.convt3 = torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.convt4 = torch.nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.convt0(x.view(*x.shape, 1, 1))
        x = self.bn0(x)
        x = F.relu(x)
        x = self.convt1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.convt2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.convt3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.convt4(x)
        x = torch.tanh(x) #map to range [-1,1], see dataloader.transform
        return x


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv0 = torch.nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=True)
        self.leaky_relu_alpha = 0.2
        self.conv1 = torch.nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(128)
        self.conv2 = torch.nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(256)
        self.conv3 = torch.nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(512)
        self.conv4 = torch.nn.Conv2d(512, 1, 4, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv0(x)
        x = F.leaky_relu(x, self.leaky_relu_alpha)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, self.leaky_relu_alpha)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, self.leaky_relu_alpha)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, self.leaky_relu_alpha)
        x = self.conv4(x).view(x.shape[0])
        # x = F.sigmoid(x)
        return x


batch_size = 128
device = torch.device('cuda:0')


tmp0 = torchvision.transforms.Compose([
        torchvision.transforms.Resize(64),
        torchvision.transforms.CenterCrop(64),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
dataset = torchvision.datasets.ImageFolder(root='data/celeba', transform=tmp0)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, workers=2) #num_workers=workers

# show_image(next(iter(dataloader))[0])


# TODO check convtranspose2d
netG = Generator().to(device)
netD = Discriminator().to(device)
fixed_noise = torch.randn(64, 100, device=device) #TODO add animation

optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

hf_loss = F.binary_cross_entropy_with_logits

metric_history = defaultdict(list)
netG.train()
netD.train()
for ind_epoch in range(10):
    with tqdm(total=len(dataloader), desc='epoch-{}'.format(ind_epoch)) as pbar:
        for ind_batch,(image_real_i,_) in enumerate(dataloader):
            image_real_i = image_real_i.to(device)
            noise_i = torch.randn(batch_size, 100, device=device)
            image_fake_i = netG(noise_i)

            # optimizerD step
            netD.zero_grad()
            logits_real = netD(image_real_i)
            logits_fake = netD(image_fake_i.detach())
            loss_D = hf_loss(logits_real, torch.ones_like(logits_real)) + hf_loss(logits_fake, torch.zeros_like(logits_fake))
            loss_D.backward()
            optimizerD.step()

            # optimizerG step
            netG.zero_grad()
            logits_fake = netD(image_fake_i)
            loss_G = hf_loss(logits_fake, torch.ones_like(logits_fake))
            loss_G.backward()
            optimizerG.step()

            metric_history['loss-D'].append(loss_D.item())
            metric_history['loss-G'].append(loss_G.item())
            pbar.set_postfix({'lossD':'{:5.3}'.format(loss_D.item()), 'lossG':'{:4.3}'.format(loss_G.item())})
            pbar.update()


# TODO show animation on fixed noise, see https://matplotlib.org/gallery/animation/dynamic_image2.html
