import numpy as np
import torch

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


def test_torch_nn_RNN_sequence(N0=7, C0=3, C1=5):
    np_x_in = np.random.rand(N0,C0).astype(np.float32)
    np_h_in = np.random.rand(C1).astype(np.float32)

    rnn = torch.nn.RNN(C0, C1)
    torch_x_out,torch_h_out = rnn(torch.tensor(np_x_in).view(N0,1,C0), torch.tensor(np_h_in).view(1,1,C1))
    # (time_sequence, batch, feature)

    tmp0 = list(rnn.parameters())
    para = {
        'w_x': tmp0[0].detach().numpy(),
        'w_h': tmp0[1].detach().numpy(),
        'b_x': tmp0[2].detach().numpy(),
        'b_h': tmp0[3].detach().numpy(),
    }
    np_x_out = [None]*N0
    np_h_out = np_h_in
    for ind0 in range(N0):
        np_h_out = np.tanh(para['w_x'] @ np_x_in[ind0] + para['w_h'] @ np_h_out + para['b_x'] + para['b_h'])
        np_x_out[ind0] = np_h_out
    np_x_out = np.stack(np_x_out)[:,np.newaxis]
    np_h_out = np_h_out[np.newaxis,np.newaxis]

    assert hfe(np_x_out, torch_x_out.detach().numpy()) < 1e-4
    assert hfe(np_h_out, torch_h_out.detach().numpy()) < 1e-4


# TODO clean-up
def test_torch_nn_lstm(C0=3, C1=5):
    x_in = np.random.rand(C0).astype(np.float32)
    h0_ = np.random.rand(C1).astype(np.float32)
    c0_ = np.random.rand(C1).astype(np.float32)
    hf_sigmoid = lambda x: 1/(1+np.exp(-x))

    lstm = torch.nn.LSTM(C0, C1)
    tmp1 = torch.tensor(x_in[np.newaxis,np.newaxis])
    h0 = torch.tensor(h0_[np.newaxis,np.newaxis])
    c0 = torch.tensor(c0_[np.newaxis,np.newaxis])
    x_out_torch,(h0_torch,c0_torch) = lstm(tmp1, (h0,c0))

    w_x,w_h,b_x,b_h = list(lstm.parameters())
    w_ii,w_if,w_ig,w_io = np.split(w_x.detach().numpy(), 4, axis=0)
    b_ii,b_if,b_ig,b_io = np.split(b_x.detach().numpy(), 4, axis=0)
    w_hi,w_hf,w_hg,w_ho = np.split(w_h.detach().numpy(), 4, axis=0)
    b_hi,b_hf,b_hg,b_ho = np.split(b_h.detach().numpy(), 4, axis=0)
    h0 = h0_
    c0 = c0_
    i_t = hf_sigmoid(np.matmul(w_ii,x_in) + b_ii + np.matmul(w_hi,h0) + b_hi)
    f_t = hf_sigmoid(np.matmul(w_if,x_in) + b_if + np.matmul(w_hf,h0) + b_hf)
    g_t = np.tanh(np.matmul(w_ig,x_in) + b_ig + np.matmul(w_hg,h0) + b_hg)
    o_t = hf_sigmoid(np.matmul(w_io,x_in) + b_io + np.matmul(w_ho,h0) + b_ho)
    c0_np = f_t*c0 + i_t*g_t
    h0_np = o_t*np.tanh(c0_np)
    x_out_np = h0_np

    assert hfe(h0_torch.detach().numpy()[0,0], h0_np) < 1e-4
    assert hfe(c0_torch.detach().numpy()[0,0], c0_np) < 1e-4
    assert hfe(x_out_torch.detach().numpy()[0,0], x_out_np) < 1e-4


# TODO clean-up
def test_torch_nn_lstm_sequence(N0=7, C0=3, C1=5):
    x_in = [np.random.rand(1,1,C0).astype(np.float32) for _ in range(N0)]
    h0_ = np.random.rand(1,1,C1).astype(np.float32)
    c0_ = np.random.rand(1,1,C1).astype(np.float32)
    lstm = torch.nn.LSTM(C0, C1)
    hf_sigmoid = lambda x: 1/(1+np.exp(-x))

    h0_1 = torch.tensor(h0_)
    c0_1 = torch.tensor(c0_)
    x_out1 = [None]*N0
    for ind1 in range(N0):
        x_out1[ind1],(h0_1,c0_1) = lstm(torch.tensor(x_in[ind1]), (h0_1,c0_1))

    h0_2 = torch.tensor(h0_)
    c0_2 = torch.tensor(c0_)
    tmp1 = torch.tensor(np.concatenate(x_in, axis=0))
    x_out2,(h0_2,c0_2) = lstm(tmp1, (h0_2,c0_2))

    x_out1_np = np.concatenate([x.detach().numpy() for x in x_out1], axis=0)
    x_out2_np = x_out2.detach().numpy()
    assert hfe(x_out1_np, x_out2_np) < 1e-4
    assert hfe(h0_1.detach().numpy(), h0_2.detach().numpy()) < 1e-4
    assert hfe(c0_1.detach().numpy(), c0_2.detach().numpy()) < 1e-4
