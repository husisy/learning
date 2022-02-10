# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
import os
import numpy as np
from tqdm import tqdm

import torch
import torchtext

class TransformerModel(torch.nn.Module):

    def __init__(self, vocab_size, emb_size, nhead, nhid, nlayers, dropout=0.5, maxlen=5000):
        super().__init__()
        self.model_type = 'Transformer'

        tmp0 = np.exp(-np.arange(0, emb_size, 2)*np.log(10000) / emb_size)
        tmp1 = np.arange(maxlen)[:,np.newaxis] * tmp0
        pos_embedding = np.stack([np.sin(tmp1),np.cos(tmp1)], axis=2).reshape(maxlen, 1, emb_size)
        self.register_buffer('pos_embedding', torch.tensor(pos_embedding, dtype=torch.float32))
        self.dropout = torch.nn.Dropout(dropout)

        encoder_layers = torch.nn.TransformerEncoderLayer(emb_size, nhead, nhid, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.src_tok_emb = torch.nn.Embedding(vocab_size, emb_size)
        self.decoder = torch.nn.Linear(emb_size, vocab_size)
        self.emb_factor = np.sqrt(emb_size)

        self.init_weights()

    def init_weights(self):
        initrange = 1/np.sqrt(self.src_tok_emb.weight.shape[1])
        self.src_tok_emb.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        N0 = src.shape[0]
        device = self.decoder.weight.device
        tmp0 = torch.triu(torch.ones((N0,N0), dtype=torch.bool, device=device), 1)
        src_mask = torch.zeros((N0,N0), dtype=torch.float32, device=device)
        src_mask.masked_fill_(tmp0, float('-inf'))

        src = self.dropout(self.src_tok_emb(src)*self.emb_factor + self.pos_embedding[:src.shape[0]])
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


TORCHTEXT_ROOT = os.path.expanduser('~/torchtext_data')

train_iter = torchtext.datasets.WikiText2(split='train', root=TORCHTEXT_ROOT)
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
tmp0 = (tokenizer(x) for x in train_iter)
vocab = torchtext.vocab.build_vocab_from_iterator(tmp0, specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

device = torch.device('cuda')

chunk_size = 20
batch_size = 35
hf0 = lambda x,y,device: torch.utils.data.TensorDataset(torch.tensor(x[:((len(x)//y)*y)], dtype=torch.int64, device=device).view(-1, y))
wikitext2_train = list(torchtext.datasets.WikiText2(split='train', root=TORCHTEXT_ROOT))
ds_train_idx = [y for x in wikitext2_train for y in vocab(tokenizer(x))]
ds_train = hf0(ds_train_idx, chunk_size, device)
dl_train = torch.utils.data.DataLoader(ds_train, shuffle=True, batch_size=batch_size)

wikitext2_val = list(torchtext.datasets.WikiText2(split='valid', root=TORCHTEXT_ROOT))
ds_val_idx = [y for x in wikitext2_val for y in vocab(tokenizer(x))]
ds_val = hf0(ds_val_idx, chunk_size, device)
dl_val = torch.utils.data.DataLoader(ds_val, shuffle=False, batch_size=batch_size)

parameter = {
    'vocab_size': len(vocab),
    'emb_size': 200,
    'nhead': 2,
    'nhid': 200,
    'nlayers': 2,
    'dropout': 0.2,
}
model = TransformerModel(**parameter).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

for epoch in range(3):
    model.train()
    total_loss = 0
    loss_history = []
    with tqdm(dl_train) as pbar:
        for data, in pbar:
            data = data.transpose(1,0)
            src,dst = data[:-1], data[1:]
            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output.view(-1, output.shape[2]), dst.reshape(-1))
            loss.backward()
            loss_history.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            pbar.set_postfix(train_loss='{:5.3}'.format(sum(loss_history[-10:])/10))
    scheduler.step()
    tmp0 = np.mean(np.array(loss_history))
    print(f'[epoch={epoch}] train_loss={tmp0:.3f}, train_ppl={np.exp(tmp0):.3f}')

    model.eval()
    loss_history = []
    with torch.no_grad():
        with tqdm(dl_val) as pbar:
            for data, in pbar:
                data = data.transpose(1,0)
                src,dst = data[:-1], data[1:]
                output = model(src)
                loss = criterion(output.view(-1, output.shape[2]), dst.reshape(-1))
                loss_history.append(loss.item())
                pbar.set_postfix(val_loss='{:5.3}'.format(sum(loss_history[-10:])/10))
    tmp0 = np.mean(np.array(loss_history))
    print(f'[epoch={epoch}] val_loss={tmp0:.3f}, val_ppl={np.exp(tmp0):.3f}')

'''
100%|█████| 2929/2929 [00:41<00:00, 70.09it/s, train_loss=5.97]
[epoch=0] train_loss=6.323, train_ppl=557.028
100%|█████| 307/307 [00:01<00:00, 206.20it/s, val_loss=5.75]
[epoch=0] val_loss=5.665, val_ppl=288.587
100%|█████| 2929/2929 [00:42<00:00, 68.60it/s, train_loss=5.59]
[epoch=1] train_loss=5.717, train_ppl=304.126
100%|█████| 307/307 [00:01<00:00, 221.69it/s, val_loss=5.6]
[epoch=1] val_loss=5.512, val_ppl=247.751
100%|█████| 2929/2929 [00:40<00:00, 73.00it/s, train_loss=5.51]
[epoch=2] train_loss=5.520, train_ppl=249.554
100%|█████| 307/307 [00:01<00:00, 219.07it/s, val_loss=5.52]
[epoch=2] val_loss=5.437, val_ppl=229.737
'''
