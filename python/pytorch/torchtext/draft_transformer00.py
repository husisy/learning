# https://pytorch.org/tutorials/beginner/translation_transformer.html
import os
import time
import functools
import torch
import torchtext
import numpy as np
from tqdm import tqdm

class Seq2SeqTransformer(torch.nn.Module):
    def __init__(self, special_symbols_idx, num_encoder_layers, num_decoder_layers, emb_size, nhead,
                src_vocab_size, dst_vocab_size, dim_feedforward=512, dropout=0.1, maxlen=5000):
        super().__init__()
        self.special_symbols_idx = special_symbols_idx
        self.transformer = torch.nn.Transformer(d_model=emb_size, nhead=nhead, num_encoder_layers=num_encoder_layers,
                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.generator = torch.nn.Linear(emb_size, dst_vocab_size)
        self.src_tok_emb = torch.nn.Embedding(src_vocab_size, emb_size)
        self.emb_factor = np.sqrt(emb_size)
        self.dst_tok_emb = torch.nn.Embedding(dst_vocab_size, emb_size)

        tmp0 = np.exp(-np.arange(0, emb_size, 2)*np.log(10000) / emb_size)
        tmp1 = np.arange(maxlen)[:,np.newaxis] * tmp0
        pos_embedding = np.stack([np.sin(tmp1),np.cos(tmp1)], axis=2).reshape(maxlen, 1, emb_size)
        self.register_buffer('pos_embedding', torch.tensor(pos_embedding, dtype=torch.float32))
        self.dropout = torch.nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        initrange = 1/np.sqrt(self.src_tok_emb.weight.shape[1])
        self.src_tok_emb.weight.data.uniform_(-initrange, initrange)
        self.dst_tok_emb.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, dst):
        N0 = dst.shape[0]
        device = self.src_tok_emb.weight.device
        tmp0 = torch.triu(torch.ones((N0,N0), dtype=torch.bool, device=device), 1)
        dst_mask = torch.zeros((N0,N0), dtype=torch.float32, device=device)
        dst_mask.masked_fill_(tmp0, float('-inf'))

        src_mask = torch.zeros((src.shape[0], src.shape[0]), device=src.device, dtype=torch.bool)

        src_padding_mask = (src==self.special_symbols_idx['<pad>']).transpose(0, 1)
        dst_padding_mask = (dst==self.special_symbols_idx['<pad>']).transpose(0, 1)

        src_emb = self.dropout(self.src_tok_emb(src)*self.emb_factor + self.pos_embedding[:src.shape[0]])
        dst_emb = self.dropout(self.dst_tok_emb(dst)*self.emb_factor + self.pos_embedding[:dst.shape[0]])
        outs = self.transformer(src_emb, dst_emb, src_mask, dst_mask, None, src_padding_mask, dst_padding_mask, src_padding_mask)
        ret = self.generator(outs)
        return ret

    def translate_one(self, src_idx, max_len):
        device = self.src_tok_emb.weight.device
        tmp0 = [self.special_symbols_idx['<bos>'],*src_idx,self.special_symbols_idx['<eos>']]
        src = torch.tensor(tmp0, dtype=torch.int64, device=device).view(-1,1)
        src_mask = torch.zeros(src.shape[0], src.shape[0], dtype=torch.bool, device=device)
        src_emb = self.dropout(self.src_tok_emb(src)*self.emb_factor + self.pos_embedding[:src.shape[0]])
        memory = self.transformer.encoder(src_emb, src_mask)
        dst_idx = []
        for _ in range(max_len-1):
            dst = torch.tensor([self.special_symbols_idx['<bos>'],*dst_idx], dtype=torch.int64, device=device).view(-1,1)
            dst_mask = torch.triu(torch.ones((dst.shape[0],dst.shape[0]), dtype=torch.bool, device=device), 1)
            dst_emb = self.dropout(self.dst_tok_emb(dst)*self.emb_factor + self.pos_embedding[:dst.shape[0]])
            tmp0 = self.transformer.decoder(dst_emb, memory, dst_mask).transpose(0, 1)
            prob = self.generator(tmp0[:,-1])
            next_word = torch.max(prob, dim=1)[1].item()
            if next_word==self.special_symbols_idx['<eos>']:
                break
            dst_idx.append(next_word)
        return dst_idx


def hf_collate(data_batch, BOS_IDX, EOS_IDX, PAD_IDX):
    data_batch = list(data_batch)
    de_batch = [torch.tensor([BOS_IDX]+x+[EOS_IDX], dtype=torch.int64) for x,_ in data_batch]
    de_batch = torch.nn.utils.rnn.pad_sequence(de_batch, padding_value=PAD_IDX)
    en_batch = [torch.tensor([BOS_IDX]+x+[EOS_IDX], dtype=torch.int64) for _,x in data_batch]
    en_batch = torch.nn.utils.rnn.pad_sequence(en_batch, padding_value=PAD_IDX)
    return de_batch, en_batch



batch_size = 128
TORCHTEXT_ROOT = os.path.expanduser('~/torchtext_data')

special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
src_tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='de_core_news_sm')
dst_tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='en_core_web_sm')
ds_Multi30k_train = list(torchtext.datasets.Multi30k(split='train', language_pair=('de','en'), root=TORCHTEXT_ROOT))
ds_Multi30k_val = list(torchtext.datasets.Multi30k(split='valid', language_pair=('de','en'), root=TORCHTEXT_ROOT))

tmp0 = (src_tokenizer(x) for x,_ in ds_Multi30k_train)
src_vocab = torchtext.vocab.build_vocab_from_iterator(tmp0, min_freq=1, specials=special_symbols, special_first=True)
special_symbols_idx = {x:src_vocab([x])[0] for x in special_symbols}
src_vocab.set_default_index(special_symbols_idx['<unk>'])
tmp0 = (dst_tokenizer(x) for _,x in ds_Multi30k_train)
dst_vocab = torchtext.vocab.build_vocab_from_iterator(tmp0, min_freq=1, specials=special_symbols, special_first=True)
assert all(dst_vocab([x])[0]==y for x,y in special_symbols_idx.items())
dst_vocab.set_default_index(special_symbols_idx['<unk>'])

ds_train_idx = [(src_vocab(src_tokenizer(x)), dst_vocab(dst_tokenizer(y))) for x,y in ds_Multi30k_train]
ds_val_idx = [(src_vocab(src_tokenizer(x)), dst_vocab(dst_tokenizer(y))) for x,y in ds_Multi30k_val]
tmp0 = dict(BOS_IDX=special_symbols_idx['<bos>'], EOS_IDX=special_symbols_idx['<eos>'], PAD_IDX=special_symbols_idx['<pad>'])
hf0 = functools.partial(hf_collate, **tmp0)
dl_train = torch.utils.data.DataLoader(ds_train_idx, batch_size=batch_size, shuffle=True, collate_fn=hf0, num_workers=2)
dl_val = torch.utils.data.DataLoader(ds_val_idx, batch_size=batch_size, shuffle=False, collate_fn=hf0, num_workers=2)


device = torch.device('cuda')

parameter = {
    'special_symbols_idx': special_symbols_idx,
    'num_encoder_layers': 3,
    'num_decoder_layers': 3,
    'emb_size': 512,
    'nhead': 8,
    'src_vocab_size': len(src_vocab),
    'dst_vocab_size': len(dst_vocab),
    'dim_feedforward': 512,
    'dropout': 0.1,
}
model = Seq2SeqTransformer(**parameter).to(device)

hf_loss = torch.nn.CrossEntropyLoss(ignore_index=special_symbols_idx['<pad>'])
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


for epoch in range(18):
    model.train()
    train_loss = 0
    start_time = time.time()
    for src,dst in dl_train:
        src,dst = src.to(device),dst.to(device)
        logits = model(src, dst[:-1])
        optimizer.zero_grad()
        loss = hf_loss(logits.reshape(-1, logits.shape[-1]), dst[1:].reshape(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss/len(dl_train)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for src, dst in dl_val:
            src,dst = src.to(device),dst.to(device)
            logits = model(src, dst[:-1])
            loss = hf_loss(logits.reshape(-1, logits.shape[-1]), dst[1:].reshape(-1))
            val_loss += loss.item()
    val_loss = val_loss / len(dl_val)
    print(f"[epoch={epoch}][{time.time()-start_time:.1f} seconds] train_loss={train_loss:.3f}, val_loss={val_loss:.3f}")


tmp0 = [dst_vocab.lookup_tokens(model.translate_one(x, max_len=50)) for x,_ in tqdm(ds_val_idx)] #around 1 minute
tmp1 = [(dst_tokenizer(x),) for _,x in ds_Multi30k_val]
bleu = torchtext.data.metrics.bleu_score(tmp0, tmp1) #around 0.39
print(f'bleu(validation): {bleu}')


src_sentence = 'Eine Gruppe von Menschen steht vor einem Iglu .'
model.eval()
src_idx = src_vocab(src_tokenizer(src_sentence))
dst_idx = model.translate_one(src_idx, max_len=len(src_idx)+7)
print(' '.join(dst_vocab.lookup_tokens(dst_idx)))

'''
[epoch=0][32.6 seconds] train_loss=4.722, val_loss=3.586
[epoch=1][32.6 seconds] train_loss=3.316, val_loss=2.907
[epoch=2][32.0 seconds] train_loss=2.803, val_loss=2.570
[epoch=3][32.4 seconds] train_loss=2.477, val_loss=2.359
[epoch=4][32.9 seconds] train_loss=2.239, val_loss=2.210
[epoch=5][32.3 seconds] train_loss=2.046, val_loss=2.115
[epoch=6][32.0 seconds] train_loss=1.884, val_loss=2.028
[epoch=7][32.3 seconds] train_loss=1.747, val_loss=1.959
[epoch=8][33.3 seconds] train_loss=1.625, val_loss=1.901
[epoch=9][33.0 seconds] train_loss=1.519, val_loss=1.869
[epoch=10][33.0 seconds] train_loss=1.417, val_loss=1.826
[epoch=11][33.4 seconds] train_loss=1.325, val_loss=1.812
[epoch=12][33.2 seconds] train_loss=1.242, val_loss=1.795
[epoch=13][33.0 seconds] train_loss=1.164, val_loss=1.769
[epoch=14][32.9 seconds] train_loss=1.092, val_loss=1.754
[epoch=15][33.5 seconds] train_loss=1.022, val_loss=1.764
[epoch=16][33.1 seconds] train_loss=0.958, val_loss=1.748
[epoch=17][33.0 seconds] train_loss=0.897, val_loss=1.752
bleu(validation): 0.3856119807581984
A group of people stand outside an amusement park .
'''
