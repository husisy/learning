## https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html
import os
import time
import tempfile
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

hf_data = lambda *x: os.path.join('..', 'data', *x)
# https://s3.amazonaws.com/pytorch-tutorial-assets/wikitext-2.zip
# https://s3.amazonaws.com/pytorch-tutorial-assets/word_language_model_quantize.pth

def get_model_file_size(model):
    state_dict = model.state_dict()
    # num_parameter = sum(x.numel() for x in state_dict.values())

    # tmp0 = sorted([(str(x.dtype).rsplit('.',1)[1], x.numel()) for x in state_dict.values()], key=lambda x:x[0])
    # dtype_to_byte = {'float32':4, 'float64':8, 'int32':4, 'int64':8}
    # size = sum([dtype_to_byte[x0]*sum(y[1] for y in x1) for x0,x1 in itertools(tmp0, key=lambda x:x[0])])

    z0 = tempfile.TemporaryDirectory()
    filepath = os.path.join(z0.name, 'model.pt')
    torch.save(state_dict, filepath)
    size_in_MB = os.path.getsize(filepath)/2**20
    os.remove(filepath)
    z0.cleanup()
    return size_in_MB

def mytimeit(hf0, num_repeat=100, is_inference=True):
    torch.set_num_threads(1) #quantized models run single threaded
    t_list = np.zeros(num_repeat+2, dtype=np.float64)
    for ind0 in range(num_repeat+2):
        t0 = time.time()
        if is_inference:
            with torch.no_grad():
                hf0()
        else:
            hf0()
        t_list[ind0] = time.time() - t0
    ret = (t_list.sum() - t_list.max() - t_list.min()) / num_repeat
    return ret


model_dimension = 8
sequence_length = 20
batch_size = 1
lstm_depth = 1

class MyModel00(torch.nn.Module):
    def __init__(self, in_dim, out_dim, depth):
        super().__init__()
        self.lstm = torch.nn.LSTM(in_dim, out_dim, depth)
    def forward(self, inputs, hidden):
        out,hidden = self.lstm(inputs, hidden)
        return out,hidden


model_fp32 = MyModel00(model_dimension, model_dimension, lstm_depth)
model_quantized = torch.quantization.quantize_dynamic(model_fp32, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8)

print('model_fp32: {:.4} MB'.format(get_model_file_size(model_fp32)))
print('model_quantized: {:.4} MB'.format(get_model_file_size(model_quantized)))
# model_fp32: 0.00357 MB
# model_quantized: 0.002593 MB

inputs = torch.randn(sequence_length,batch_size,model_dimension)
hidden = torch.randn(lstm_depth,batch_size,model_dimension), torch.randn(lstm_depth,batch_size,model_dimension)
# out1, hidden1 = model_fp32(inputs, hidden)


for is_inference in [False,True]:
    t_fp32 = mytimeit(lambda: model_fp32(inputs, hidden), is_inference=is_inference)
    # %timeit model_fp32(inputs, hidden)
    t_quantized = mytimeit(lambda: model_quantized(inputs, hidden), is_inference=is_inference)
    # %timeit mytimeit(inputs, hidden)
    print(f'model_fp32(is_inference={is_inference}): {t_fp32*1000:.3} ms')
    print(f'model_quantized(is_inference={is_inference}): {t_quantized*1000:.3} ms')
# model_fp32(is_inference=False): 0.746 ms
# model_quantized(is_inference=False): 0.314 ms
# model_fp32(is_inference=True): 0.284 ms
# model_quantized(is_inference=True): 0.326 ms


out_fp32 = model_fp32(inputs, hidden)[0].detach().numpy().reshape(-1)
out_quantized = model_quantized(inputs, hidden)[0].numpy().reshape(-1)
print('model_fp32():', out_fp32[:4])
print('model_quantized():', out_quantized[:4])


## https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html
class MyModel01(torch.nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super().__init__()
        self.drop = torch.nn.Dropout(dropout)
        self.encoder = torch.nn.Embedding(ntoken, ninp)
        self.rnn = torch.nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = torch.nn.Linear(nhid, ntoken)
        self.nhid = nhid
        self.nlayers = nlayers
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        x = self.drop(self.encoder(input))
        x, hidden = self.rnn(x, hidden)
        x = self.decoder(self.drop(x))
        return x, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        ret = [weight.new_zeros(self.nlayers, batch_size, self.nhid) for _ in range(2)]
        return ret



class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Wiki2Corpus:
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        with open(path, 'r', encoding="utf8") as fid:
            z0 = [x.split()+['<eos>'] for x in fid]
        for x in z0:
            for y in x:
                self.dictionary.add_word(y)
        ids = torch.tensor([self.dictionary.word2idx[y] for x in z0 for y in x], dtype=torch.int64)
        return ids

def split_by_item(item_list, item):
    ret = []
    tmp0 = [x for x,y in enumerate(item_list) if y==item] + [len(item_list)]
    ret = [item_list[:tmp0[0]]] + [item_list[(x+1):y] for x,y in zip(tmp0[:-1],tmp0[1:])]
    return ret


corpus = Wiki2Corpus(hf_data('wikitext-2'))
ntokens = len(corpus.dictionary)
model = MyModel01(ntoken=ntokens, ninp=512, nhid=256, nlayers=5)
model.load_state_dict(torch.load(hf_data('word_language_model_quantize.pth'), map_location=torch.device('cpu')))

input_ = torch.randint(ntokens, size=(1,1), dtype=torch.int64)
hidden = model.init_hidden(1)
temperature = 1.0
num_words = 1000
all_output = []
model.eval()
with torch.no_grad():
    for _ in range(num_words):
        output, hidden = model(input_, hidden)
        word_weights = output.squeeze().div(temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        input_.fill_(word_idx)
        all_output.append(corpus.dictionary.idx2word[word_idx])
for x in split_by_item(all_output, '<eos>'):
    print(' '.join(x), '\n')


def evaluate(model, test_data, bptt):
    torch.set_num_threads(1) #quantized models run single threaded
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(batch_size=1)
    with torch.no_grad():
        for i in tqdm(range(0, test_data.size(0) - 1, bptt)):
            seq_len = min(bptt, len(test_data)- 1 - i)
            data = test_data[i:i+seq_len]
            target = test_data[i+1:i+1+seq_len].reshape(-1)
            # data, targets = get_batch(test_data, i, bptt)
            output, hidden = model(data, hidden)
            total_loss += len(data) * F.cross_entropy(output[:,0], targets).item()
    ret = total_loss / (len(test_data) - 1)
    return ret

bptt = 25 #backpropagation through time
test_data = corpus.test.view(-1, 1)
model_quantized = torch.quantization.quantize_dynamic(model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8)
print('loss(model):', evaluate(model, test_data, bptt))
print('loss(model_quantized):', evaluate(model_quantized, test_data, bptt))
# loss(model): 5.167401954508854
# loss(model_quantized): 5.167555265782019
