import numpy as np
import torch

assert torch.cuda.is_available()

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
to_np = lambda x,dtype=np.float32: x.cpu().detach().numpy().astype(dtype)
hfe16 = lambda x,y,eps=1e-3: hfe(x.astype(np.float32), y.astype(np.float32), eps)

def _autocast_dtype():
    device = torch.device('cuda')
    torch0_f32 = torch.rand(3, 3, dtype=torch.float32, device=device)
    torch0_f16 = torch.rand(3, 3, dtype=torch.float16, device=device)

    default = torch.rand(3, device=device).dtype
    f32_x_f32 = torch.matmul(torch0_f32, torch0_f32).dtype
    f16_x_f16 = torch.matmul(torch0_f16, torch0_f16).dtype
    try:
        f32_x_f16 = torch.matmul(torch0_f32, torch0_f16).dtype
    except RuntimeError:
        f32_x_f16 = 'RuntimeError'
    ret = {
        'default': default,
        'f32_x_f32': f32_x_f32,
        'f16_x_f16': f16_x_f16,
        'f32_x_f16': f32_x_f16,
    }
    return ret


def demo_autocast():
    print('#general scope\n', _autocast_dtype())
    # default: torch.float32
    # f32_x_f32: torch.float32
    # f16_x_f16: torch.float16
    # f32_x_f16: RuntimeError

    with torch.cuda.amp.autocast():
        print('amp.autocast() scope:', _autocast_dtype())
        # default: torch.float32
        # f32_x_f32: torch.float16
        # f16_x_f16: torch.float16
        # f32_x_f16: torch.float16

        with torch.cuda.amp.autocast(enabled=False):
            print('amp.autocast(enabled=False) scope:', _autocast_dtype())
            # default: torch.float32
            # f32_x_f32: torch.float32
            # f16_x_f16: torch.float16
            # f32_x_f16: RuntimeError


def demo_strange_nn(N1=5, N2=7):
    np_rng = np.random.default_rng()
    np0 = np_rng.normal(size=(N1, N2)).astype(np.float16)
    torch0 = torch.tensor(np0, dtype=torch.float16, device='cuda')
    net_bn = torch.nn.BatchNorm1d(N2).cuda()
    net_ln = torch.nn.LayerNorm(N2).cuda()
    net_linear = torch.nn.Linear(N1, N2).cuda()
    with torch.cuda.amp.autocast():
        for net in [net_bn,net_ln,net_linear]:
            tmp0 = str(net(torch0).dtype)[6:]
            tmp1 = str(net(torch0.float()).dtype)[6:]
            print(f'{type(net).__name__}(float16)={tmp0}, {type(net).__name__}(float32)={tmp1}')
    # BatchNorm1d(float16)=float16, BatchNorm1d(float32)=float32
    # LayerNorm(float16)=float32, LayerNorm(float32)=float32
    # Linear(float16)=float16, Linear(float32)=float16
