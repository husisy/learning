import os
import torch
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

TORCH_DATADIR = os.path.expanduser('~/torch_data')

class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(28 * 28, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(3, 64), torch.nn.ReLU(), torch.nn.Linear(64, 28 * 28))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        ret = self.encoder(x)
        return ret

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss) #Logging to TensorBoard by default
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


dataset = torchvision.datasets.MNIST(TORCH_DATADIR, transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset, batch_size=128)

autoencoder = LitAutoEncoder()

# most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
# trainer = pl.Trainer(gpus=8) (if you have GPUs)
trainer = pl.Trainer(gpus=1)
trainer.fit(autoencoder, train_loader)
