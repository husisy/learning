# https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
import os
import math
import time
import random
import torch
import torchtext
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = torch.nn.Embedding(input_dim, emb_dim)
        self.rnn = torch.nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        self.fc = torch.nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden

class Attention(torch.nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, attn_dim):
        super().__init__()
        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        self.attn = torch.nn.Linear(self.attn_in, attn_dim)

    def forward(self, decoder_hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        tmp0 = torch.cat((repeated_decoder_hidden, encoder_outputs), dim = 2)
        energy = torch.tanh(self.attn(tmp0))
        attention = torch.sum(energy, dim=2)
        return F.softmax(attention, dim=1)

class Decoder(torch.nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.attention = attention
        self.embedding = torch.nn.Embedding(output_dim, emb_dim)
        self.rnn = torch.nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.out = torch.nn.Linear(self.attention.attn_in + emb_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def _weighted_encoder_rep(self, decoder_hidden, encoder_outputs):
        a = self.attention(decoder_hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted_encoder_rep = torch.bmm(a, encoder_outputs)
        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)
        return weighted_encoder_rep

    def forward(self, input, decoder_hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden, encoder_outputs)
        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim = 2)
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)
        output = self.out(torch.cat((output, weighted_encoder_rep, embedded), dim = 1))
        return output, decoder_hidden.squeeze(0)

class Seq2Seq(torch.nn.Module):
    def __init__(self, input_dim, enc_emb_dim, dec_emb_dim, enc_hid_dim,
                    dec_hid_dim, attn_dim, enc_dropout, dec_dropout, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.encoder = Encoder(input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, enc_dropout)
        self.attn = Attention(enc_hid_dim, dec_hid_dim, attn_dim)
        self.decoder = Decoder(output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, dec_dropout, self.attn)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        max_len,batch_size = trg.shape
        outputs = torch.zeros(max_len, batch_size, self.output_dim).to(src.device)
        encoder_outputs, hidden = self.encoder(src)
        # first input to the decoder is the <sos> token
        output = trg[0,:]
        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            output = trg[t] if (random.random()<teacher_forcing_ratio) else output.max(1)[1]
        return outputs


special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
# bos: begin of sentence

def build_vocab(filepath, tokenizer):
    with open(filepath, encoding="utf8") as fid:
        tmp0 = (tokenizer(x) for x in fid)
        ret = torchtext.vocab.build_vocab_from_iterator(tmp0, specials=special_symbols)
    ret.set_default_index(ret['<unk>'])
    return ret

def _data_process_i(filepath, tokenizer, vocab):
    with open(filepath, encoding='utf8') as fid:
        ret = [torch.tensor([vocab[y] for y in tokenizer(x)], dtype=torch.int64) for x in fid]
    return ret

def data_process(src_filepath, dst_filepath, src_tokenizer, dst_tokenizer, src_vocab, dst_vocab):
    tmp0 = _data_process_i(src_filepath, src_tokenizer, src_vocab)
    tmp1 = _data_process_i(dst_filepath, dst_tokenizer, dst_vocab)
    assert len(tmp0)==len(tmp1)
    ret = list(zip(tmp0, tmp1))
    return ret


url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
train_urls = ('train.de.gz', 'train.en.gz')
val_urls = ('val.de.gz', 'val.en.gz')
test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

hf_data = lambda *x: os.path.join('data')

hf0 = lambda x,root: [torchtext.utils.extract_archive(torchtext.utils.download_from_url(url_base+y, root=root))[0] for y in x]
train_filepaths = hf0(train_urls, hf_data())
val_filepaths = hf0(val_urls, hf_data())
test_filepaths = hf0(test_urls, hf_data())

de_tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='en_core_web_sm')

de_vocab = build_vocab(train_filepaths[0], de_tokenizer)
en_vocab = build_vocab(train_filepaths[1], en_tokenizer)

train_data = data_process(*train_filepaths, de_tokenizer, en_tokenizer, de_vocab, en_vocab)
val_data = data_process(*val_filepaths, de_tokenizer, en_tokenizer, de_vocab, en_vocab)
test_data = data_process(*test_filepaths, de_tokenizer, en_tokenizer, de_vocab, en_vocab)

device = torch.device('cuda')

BATCH_SIZE = 128
UNK_IDX = de_vocab['<unk>'] #0
PAD_IDX = de_vocab['<pad>'] #1
BOS_IDX = de_vocab['<bos>'] #2
EOS_IDX = de_vocab['<eos>'] #3

def generate_batch(data_batch):
    data_batch = list(data_batch)
    de_batch = [torch.cat([torch.tensor([BOS_IDX]), x, torch.tensor([EOS_IDX])], dim=0) for x,_ in data_batch]
    de_batch = torch.nn.utils.rnn.pad_sequence(de_batch, padding_value=PAD_IDX)
    en_batch = [torch.cat([torch.tensor([BOS_IDX]), x, torch.tensor([EOS_IDX])], dim=0) for _,x in data_batch]
    en_batch = torch.nn.utils.rnn.pad_sequence(en_batch, padding_value=PAD_IDX)
    return de_batch, en_batch

train_iter = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
valid_iter = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
test_iter = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)
# TODO shuffle=True still affect the loss in test_iter which should not be the case

# parameter = {
#     'input_dim': len(de_vocab), #19215
#     'output_dim': len(en_vocab), #10838
#     'enc_emb_dim': 256,
#     'dec_emb_dim': 256,
#     'enc_hid_dim': 512,
#     'dec_hid_dim': 512,
#     'attn_dim': 64,
#     'enc_dropout': 0.5,
#     'dec_dropout': 0.5,
# }
parameter = {
    'input_dim': len(de_vocab), #19215
    'output_dim': len(en_vocab), #10838
    'enc_emb_dim': 32,
    'dec_emb_dim': 32,
    'enc_hid_dim': 64,
    'dec_hid_dim': 64,
    'attn_dim': 8,
    'enc_dropout': 0.5,
    'dec_dropout': 0.5,
}
model = Seq2Seq(**parameter).to(device)

optimizer = torch.optim.Adam(model.parameters())

num_parameter = sum(x.numel() for x in model.parameters() if x.requires_grad) #3491070

criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)


def train(model, iterator, optimizer, criterion, clip=1):
    model.train()
    epoch_loss = 0
    for src, trg in iterator:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        loss = criterion(output[1:].view(-1,output.shape[-1]), trg[1:].view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in iterator:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, 0) #turn off teacher forcing
            loss = criterion(output[1:].view(-1,output.shape[-1]), trg[1:].view(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


for epoch in range(10):
    start_time = time.time()
    train_loss = train(model, train_iter, optimizer, criterion)
    valid_loss = evaluate(model, valid_iter, criterion)
    duration_time = time.time() - start_time
    print(f'[epoch={epoch}][{duration_time:.0f} second] train_loss={train_loss:.3f}, train_ppl={math.exp(train_loss):.3f}, '
            f'val_loss={valid_loss:.3f}, val_ppl={math.exp(valid_loss):.3f}')
test_loss = evaluate(model, test_iter, criterion)
print(f'test_loss={test_loss:.3f}, test_ppl: {math.exp(test_loss):.3f}')

'''
[epoch=0][44 second] train_loss=5.276, train_ppl=195.513, val_loss=4.995, val_ppl=147.675
[epoch=1][45 second] train_loss=4.662, train_ppl=105.812, val_loss=4.835, val_ppl=125.832
[epoch=2][45 second] train_loss=4.452, train_ppl=85.840, val_loss=4.718, val_ppl=111.958
[epoch=3][46 second] train_loss=4.286, train_ppl=72.650, val_loss=4.622, val_ppl=101.702
[epoch=4][46 second] train_loss=4.120, train_ppl=61.580, val_loss=4.412, val_ppl=82.441
[epoch=5][46 second] train_loss=3.911, train_ppl=49.949, val_loss=4.200, val_ppl=66.667
[epoch=6][46 second] train_loss=3.728, train_ppl=41.606, val_loss=4.109, val_ppl=60.901
[epoch=7][46 second] train_loss=3.600, train_ppl=36.585, val_loss=4.032, val_ppl=56.380
[epoch=8][46 second] train_loss=3.490, train_ppl=32.800, val_loss=3.969, val_ppl=52.921
[epoch=9][46 second] train_loss=3.408, train_ppl=30.218, val_loss=3.922, val_ppl=50.483
test_loss=3.891, test_ppl: 48.969
'''
