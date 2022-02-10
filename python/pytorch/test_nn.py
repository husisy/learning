import numpy as np
import scipy.special
import torch
import torch.nn.functional as F

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


def test_batchnorm1d_train(num_batch=3, N0=3, N1=5, N2=7, momentum=0.233, eps=0.00233):
    np_weight = np.random.randn(N1).astype(np.float32)
    np_bias = np.random.randn(N1).astype(np.float32)
    batch_list = [np.random.randn(N0,N1,N2).astype(np.float32) for _ in range(num_batch)]

    running_mean_ = 0
    running_var_ = 1
    for batch_i in batch_list:
        running_mean_ = (1-momentum)*running_mean_ + momentum*batch_i.mean(axis=(0,2))
        running_var_ = (1-momentum)*running_var_ + momentum*batch_i.var(axis=(0,2), ddof=1)
    tmp0 = batch_list[-1]
    tmp1 = (tmp0 - tmp0.mean(axis=(0,2),keepdims=True)) / np.sqrt(tmp0.var(axis=(0,2),keepdims=True) + eps)
    ret_ = tmp1*np_weight[:,np.newaxis] + np_bias[:,np.newaxis]

    bn0 = torch.nn.BatchNorm1d(N1, momentum=momentum, eps=eps)
    bn0.weight.data = torch.tensor(np_weight)
    bn0.bias.data = torch.tensor(np_bias)
    bn0.train()
    for batch_i in batch_list:
        ret0 = bn0(torch.tensor(batch_i))
    assert hfe(ret_, ret0.detach().numpy()) < 1e-4
    assert hfe(running_mean_, bn0.running_mean.detach().numpy()) < 1e-4
    assert hfe(running_var_, bn0.running_var.detach().numpy()) < 1e-4


def test_batchnorm1d_eval(N0=3, N1=5, N2=7, eps=0.00233):
    running_mean = np.random.randn(N1).astype(np.float32)
    running_var = np.random.uniform(1, 2, size=(N1,)).astype(np.float32)
    np0 = np.random.randn(N0, N1, N2).astype(np.float32)
    np_weight = np.random.randn(N1).astype(np.float32)
    np_bias = np.random.randn(N1).astype(np.float32)

    tmp0 = (np0 - running_mean[:,np.newaxis]) / np.sqrt(running_var[:,np.newaxis]+eps)
    ret_ = tmp0 * np_weight[:,np.newaxis] + np_bias[:,np.newaxis]

    bn0 = torch.nn.BatchNorm1d(N1, eps=eps)
    bn0.weight.data = torch.tensor(np_weight)
    bn0.bias.data = torch.tensor(np_bias)
    bn0.running_mean = torch.tensor(running_mean)
    bn0.running_var = torch.tensor(running_var)
    bn0.eval()
    ret0 = bn0(torch.tensor(np0))
    assert hfe(ret_, ret0.detach().numpy()) < 1e-4


# NEVER use track_running_stats=False, also see https://www.zhihu.com/question/282672547
def test_batchnorm1d_track_running_stats(N0=3, N1=5, eps=0.000233):
    np0 = np.random.randn(N0, N1).astype(np.float32)
    running_mean = np.random.randn(N1).astype(np.float32)
    running_var = np.random.uniform(1, 2, size=(N1,)).astype(np.float32)

    ret_eval_good = (np0 - running_mean) / np.sqrt(running_var+eps)
    ret_eval_bad = (np0 - np0.mean(axis=0)) / np.sqrt(np0.var(axis=0)+eps)

    bn0 = torch.nn.BatchNorm1d(N1, track_running_stats=True, eps=eps, affine=False)
    bn0.running_mean = torch.tensor(running_mean)
    bn0.running_var = torch.tensor(running_var)
    bn0.eval() #equivalent bn0.training=False
    ret0 = bn0(torch.tensor(np0)).detach().numpy()
    assert hfe(ret_eval_good, ret0) < 1e-4

    bn1 = torch.nn.BatchNorm1d(N1, track_running_stats=False, eps=eps, affine=False)
    # bn1.running_mean = torch.tensor(running_mean) #error cannot assign
    # bn1.running_var = torch.tensor(running_var) #error cannot assign
    bn1.eval()
    ret1 = bn1(torch.tensor(np0)).detach().numpy()
    assert hfe(ret_eval_bad, ret1) < 1e-4


def test_F_batch_norm(N0=3, N1=5, eps=0.0233):
    shape = N0,N1,3
    np0 = np.random.randn(*shape).astype(np.float32)
    ret_ = (np0 - np0.mean(axis=(0,2),keepdims=True)) / np.sqrt(np0.var(axis=(0,2),keepdims=True)+eps)
    ret0 = F.batch_norm(torch.tensor(np0), None, None, None, None, True, 0, eps).numpy()
    assert hfe(ret_, ret0) < 1e-5

    shape = N0,N1,3,3
    np0 = np.random.randn(*shape).astype(np.float32)
    ret_ = (np0 - np0.mean(axis=(0,2,3),keepdims=True)) / np.sqrt(np0.var(axis=(0,2,3),keepdims=True)+eps)
    ret0 = F.batch_norm(torch.tensor(np0), None, None, None, None, True, 0, eps).numpy()
    assert hfe(ret_, ret0) < 1e-5

    shape = N0,N1,3,3,3
    np0 = np.random.randn(*shape).astype(np.float32)
    ret_ = (np0 - np0.mean(axis=(0,2,3,4),keepdims=True)) / np.sqrt(np0.var(axis=(0,2,3,4),keepdims=True)+eps)
    ret0 = F.batch_norm(torch.tensor(np0), None, None, None, None, True, 0, eps).numpy()
    assert hfe(ret_, ret0) < 1e-5


def test_nn_utils_clip_grad_norm_(N0=3, shape=(3,1024), max_norm=23):
    np_list = [np.random.rand(*shape).astype(np.float32) for _ in range(N0)]
    np_norm = np.sqrt(sum([(x**2).sum() for x in np_list]))
    ret_ = [x*min(1, max_norm/np_norm) for x in np_list]

    torch_list = [torch.rand(*shape, requires_grad=True, dtype=torch.float32) for _ in range(N0)]
    for x,y in zip(np_list,torch_list):
        y.grad = torch.tensor(x, dtype=torch.float32)
    torch_norm = torch.nn.utils.clip_grad_norm_(torch_list, max_norm)

    assert hfe(np_norm, torch_norm.numpy()) < 1e-4
    assert max([hfe(x,y.grad.numpy()) for x,y in zip(ret_,torch_list)]) < 1e-4


def test_nn_utils_clip_grad_value_(N0=3, shape=(3,1024), clip_value=0.233):
    np_list = [np.random.randn(*shape).astype(np.float32) for _ in range(N0)]
    ret_ = [np.clip(x,-clip_value,clip_value) for x in np_list]

    torch_list = [torch.rand(*shape, requires_grad=True, dtype=torch.float32) for _ in range(N0)]
    for x,y in zip(np_list,torch_list):
        y.grad = torch.tensor(x, dtype=torch.float32)
    torch.nn.utils.clip_grad_value_(torch_list, clip_value)

    assert max([hfe(x,y.grad.numpy()) for x,y in zip(ret_,torch_list)]) < 1e-4


class DummyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = torch.nn.Linear(3, 6)
        self.dropout = torch.nn.Dropout(0.5)
    def forward(self, x):
        ret = self.dropout(self.fc0(x))
        return ret

def test_nn_dropout():
    net = DummyNet()
    torch0 = torch.randn(1, 3)
    net.train()
    z0 = np.concatenate([net(torch0).data.numpy() for _ in range(5)], axis=0)
    assert np.std(z0, axis=0).max() > 1e-3
    net.eval()
    z0 = np.concatenate([net(torch0).data.numpy() for _ in range(5)], axis=0)
    assert np.std(z0, axis=0).max() < 1e-5


def test_nn_LayerNorm(N0=3, N1=5, N2=7, eps=0.0233):
    np_rng = np.random.default_rng()
    np0 = np_rng.normal(size=(N0, N1, N2)).astype(np.float32)
    np_gamma = np_rng.normal(size=N2).astype(np.float32)
    np_beta = np_rng.normal(size=N2).astype(np.float32)

    tmp0 = (np0 - np0.mean(axis=2, keepdims=True)) / np.sqrt(np.var(np0, axis=2, ddof=0, keepdims=True) + eps)
    ret_ = tmp0*np_gamma + np_beta

    # LayerNorm doesn't depend on training=True/False
    torch0 = torch.tensor(np0, dtype=torch.float32)
    layer = torch.nn.LayerNorm(N2, eps=eps, device='cpu', dtype=torch.float32)
    layer.weight.data.copy_(torch.tensor(np_gamma, dtype=torch.float32))
    layer.bias.data.copy_(torch.tensor(np_beta, dtype=torch.float32))
    ret0 = layer(torch0).detach().numpy()
    assert hfe(ret_, ret0) < 1e-5


def multi_head_attention_forward(query, key, value, num_heads, in_proj_weight, in_proj_bias,
            out_proj_weight, out_proj_bias, key_padding_mask, attn_mask):
    q_len, batch_size, emb_size = query.shape
    k_len = key.shape[0]
    head_dim = emb_size // num_heads
    assert (emb_size % num_heads)==0
    assert key.shape == value.shape
    assert key_padding_mask.shape == (batch_size, k_len)
    tmp0,tmp1,tmp2 = in_proj_weight.chunk(3)
    tmp3,tmp4,tmp5 = in_proj_bias.chunk(3)
    q = torch.matmul(query.view(-1,emb_size), tmp0.t()).view(*query.shape) + tmp3
    k = torch.matmul(key.view(-1, emb_size), tmp1.t()).view(*key.shape) + tmp4
    v = torch.matmul(value.view(-1, emb_size), tmp2.t()).view(*value.shape) + tmp5
    q = q.contiguous().view(q_len, batch_size*num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(k_len, batch_size*num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(k_len, batch_size*num_heads, head_dim).transpose(0, 1)

    tmp0 = key_padding_mask.view(batch_size, 1, k_len).expand(-1, num_heads, -1).reshape(batch_size*num_heads, 1, k_len)
    tmp1 = torch.bmm(q/np.sqrt(head_dim), k.transpose(1, 2)) + attn_mask.masked_fill(tmp0, float("-inf"))
    tmp2 = F.softmax(tmp1, dim=2)
    attn_output_weights = tmp2.view(batch_size, num_heads, q_len, k_len).mean(dim=1)
    tmp3 = torch.bmm(tmp2, v).transpose(0, 1).contiguous().view(q_len, batch_size, emb_size)
    attn_output = torch.matmul(tmp3.view(-1,emb_size), out_proj_weight.t()).view(*tmp3.shape) + out_proj_bias
    return attn_output, attn_output_weights

def np_multi_head_attention_forward(query, key, value, num_heads, in_proj_weight, in_proj_bias,
            out_proj_weight, out_proj_bias, key_padding_mask, attn_mask):
    q_len, batch_size, emb_size = query.shape
    k_len = key.shape[0]
    head_dim = emb_size // num_heads
    assert (emb_size % num_heads)==0
    assert key.shape == value.shape
    assert key_padding_mask.shape == (batch_size, k_len)
    tmp0,tmp1,tmp2 = np.split(in_proj_weight, 3, axis=0)
    tmp3,tmp4,tmp5 = np.split(in_proj_bias, 3, axis=0)
    q = np.matmul(query, tmp0.T) + tmp3
    k = np.matmul(key, tmp1.T) + tmp4
    v = np.matmul(value, tmp2.T) + tmp5
    q = q.reshape(q_len, batch_size*num_heads, head_dim).transpose(1,0,2)
    k = k.reshape(k_len, batch_size*num_heads, head_dim).transpose(1,0,2)
    v = v.reshape(k_len, batch_size*num_heads, head_dim).transpose(1,0,2)
    tmp0 = np.tile(key_padding_mask[:,np.newaxis], (1,num_heads*q_len,1)).reshape(batch_size*num_heads,q_len,k_len)
    tmp1 = np.tile(attn_mask[np.newaxis], (batch_size*num_heads,1,1))
    tmp1[tmp0] = float('-inf')
    tmp1 = np.matmul(q/np.sqrt(head_dim), k.transpose(0,2,1)) + tmp1
    tmp2 = scipy.special.softmax(tmp1, axis=2)
    attn_output_weights = tmp2.reshape(batch_size, num_heads, q_len, k_len).mean(axis=1)
    tmp3 = np.matmul(tmp2, v).transpose(1,0,2).reshape(q_len, batch_size, emb_size)
    attn_output = np.matmul(tmp3, out_proj_weight.T) + out_proj_bias
    return attn_output, attn_output_weights

def test_multi_head_attention_forward():
    batch_size = 3
    q_len = 18
    k_len = 16
    emb_size = 24
    num_heads = 8
    assert (emb_size%num_heads)==0
    np_rng = np.random.default_rng()
    hf_randn = lambda *x: np_rng.normal(size=x).astype(np.float32)
    np_query = hf_randn(q_len, batch_size, emb_size)
    np_key = hf_randn(k_len, batch_size, emb_size)
    np_value = hf_randn(k_len, batch_size, emb_size)
    np_pad_mask = np.array([np.arange(k_len)>(np_rng.integers(int(0.5*k_len), k_len)) for _ in range(batch_size)], dtype=np.bool_)
    np_attn_mask = hf_randn(q_len, k_len)
    np_in_project_weight = hf_randn(3*emb_size, emb_size)
    np_in_project_bias = hf_randn(3*emb_size)
    np_out_project_weight = hf_randn(emb_size, emb_size)
    np_out_project_bias = hf_randn(emb_size)

    torch_query = torch.tensor(np_query, dtype=torch.float32)
    torch_key = torch.tensor(np_key, dtype=torch.float32)
    torch_value = torch.tensor(np_value, dtype=torch.float32)
    torch_attn_mask = torch.tensor(np_attn_mask, dtype=torch.float32)
    torch_in_project_weight = torch.tensor(np_in_project_weight, dtype=torch.float32)
    torch_in_project_bias = torch.tensor(np_in_project_bias, dtype=torch.float32)
    torch_out_project_weight = torch.tensor(np_out_project_weight, dtype=torch.float32)
    torch_out_project_bias = torch.tensor(np_out_project_bias, dtype=torch.float32)
    torch_pad_mask = torch.tensor(np_pad_mask, dtype=torch.bool)
    ret0_,ret1_ = F.multi_head_attention_forward(torch_query, torch_key, torch_value, emb_size, num_heads, torch_in_project_weight,
            torch_in_project_bias, bias_k=None, bias_v=None, out_proj_weight=torch_out_project_weight,
            out_proj_bias=torch_out_project_bias, add_zero_attn=False, dropout_p=0,
            key_padding_mask=torch_pad_mask, attn_mask=torch_attn_mask)
    ret0_,ret1_ = ret0_.numpy(), ret1_.numpy()

    ret0,ret1 = multi_head_attention_forward(torch_query, torch_key, torch_value, num_heads, torch_in_project_weight,
            torch_in_project_bias, torch_out_project_weight, torch_out_project_bias, torch_pad_mask, torch_attn_mask)
    ret0,ret1 = ret0.numpy(), ret1.numpy()
    assert hfe(ret0_, ret0) < 1e-5
    assert hfe(ret1_, ret1) < 1e-5

    ret2,ret3 = np_multi_head_attention_forward(np_query, np_key, np_value, num_heads, np_in_project_weight,
            np_in_project_bias, np_out_project_weight, np_out_project_bias, np_pad_mask, np_attn_mask)
    assert hfe(ret0_, ret2) < 1e-3 #sometimes large hfe due to those small number near 1e-5
    assert hfe(ret1_, ret3) < 1e-5
