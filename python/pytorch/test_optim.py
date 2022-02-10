import numpy as np
import torch

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

# LBFGS closure https://pytorch.org/docs/stable/optim.html#optimizer-step-closure
def test_torch_optim_SGD(learning_rate=0.2, num_step=5, momentum=0.23, dampening=0.233, weight_decay=0.023):
    np0 = np.random.randn(3)
    np1 = np.random.randn(3)

    np_momentum_step_i = np.zeros_like(np0)
    np0_step_i = np0
    for ind0 in range(num_step):
        np_grad_step_i = 2*np0_step_i*np1 + np0_step_i*weight_decay
        if ind0==0:
            np_momentum_step_i = np_grad_step_i
        else:
            np_momentum_step_i = momentum*np_momentum_step_i + (1-dampening)*np_grad_step_i
        np0_step_i = np0_step_i - learning_rate*np_momentum_step_i
    ret_ = np0_step_i

    torch0 = torch.tensor(np0, requires_grad=True)
    optimizer = torch.optim.SGD([torch0], lr=learning_rate,
            momentum=momentum, dampening=dampening, weight_decay=weight_decay)
    for _ in range(num_step):
        torch.sum((torch0**2)*torch.tensor(np1)).backward()
        optimizer.step()
        optimizer.zero_grad()
    ret0 = torch0.detach().numpy()
    assert hfe(ret_, ret0) < 1e-7
    tmp0 = optimizer.state_dict()['state'][0]['momentum_buffer'].numpy() #pytorch-1.6.0 change key to index
    assert hfe(np_momentum_step_i, tmp0) < 1e-7


def test_torch_optim_Adam(learning_rate=0.0233, num_step=23, beta0=0.8233, beta1=0.9233, eps=2.33e-4):
    np0 = np.random.randn(3)
    np1 = np.random.randn(3)

    np_adam_m_step_i = 0
    np_adam_v_step_i = 0
    np0_step_i = np0
    for ind0 in range(num_step):
        np_grad_step_i = 2*np0_step_i*np1
        np_adam_m_step_i = beta0*np_adam_m_step_i + (1-beta0)*np_grad_step_i
        np_adam_v_step_i = beta1*np_adam_v_step_i + (1-beta1)*np_grad_step_i**2
        tmp0 = np_adam_m_step_i / (1-beta0**(ind0+1))
        tmp1 = np_adam_v_step_i / (1-beta1**(ind0+1))
        np0_step_i = np0_step_i - learning_rate*tmp0 / (np.sqrt(tmp1) + eps)
    ret_ = np0_step_i

    torch0 = torch.tensor(np0, requires_grad=True)
    optimizer = torch.optim.Adam([torch0], lr=learning_rate, betas=(beta0,beta1), eps=eps)
    for _ in range(num_step):
        torch.sum((torch0**2)*torch.tensor(np1)).backward()
        optimizer.step()
        optimizer.zero_grad()
    ret0 = torch0.detach().numpy()
    assert hfe(ret_, ret0) < 1e-7
    tmp0 = optimizer.state_dict()['state'][0] #pytorch-1.6.0 change key to index
    assert hfe(np_adam_m_step_i, tmp0['exp_avg'].numpy()) < 1e-7
    assert hfe(np_adam_v_step_i, tmp0['exp_avg_sq'].numpy()) < 1e-7


def test_torch_optim_Adagrad(learning_rate=0.023, num_step=23, lr_decay=0.0233, eps=0.02333):
    np0 = np.random.randn(3)
    np1 = np.random.randn(3)

    np_ada_sum_i = 0
    np0_step_i = np0
    for ind0 in range(num_step):
        np_grad_step_i = 2*np0_step_i*np1
        np_ada_sum_i = np_ada_sum_i + np_grad_step_i**2
        tmp0 = learning_rate/(1+lr_decay*ind0)
        np0_step_i = np0_step_i - tmp0 * np_grad_step_i / (np.sqrt(np_ada_sum_i) + eps)
    ret_ = np0_step_i

    torch0 = torch.tensor(np0, requires_grad=True)
    optimizer = torch.optim.Adagrad([torch0], lr=learning_rate, lr_decay=lr_decay, eps=eps)
    for _ in range(num_step):
        torch.sum((torch0**2)*torch.tensor(np1)).backward()
        optimizer.step()
        optimizer.zero_grad()
    ret0 = torch0.detach().numpy()
    assert hfe(ret_, ret0) < 1e-7
    tmp0 = optimizer.state_dict()['state'][0]['sum'].numpy() #pytorch-1.6.0 change key to index
    assert hfe(np_ada_sum_i, tmp0) < 1e-7


def test_torch_optim_lr_scheduler_ExponentialLR(lr0=0.233, gamma=0.95, num_epoch=5):
    np0 = np.random.rand(3)
    np1 = np.random.rand(3)
    ret_ = np0
    lr_current = lr0
    for ind0 in range(num_epoch):
        ret_ = ret_ - lr_current*2*(ret_-np1)
        lr_current = lr_current*gamma

    torch0 = torch.tensor(np0, requires_grad=True)
    torch1 = torch.tensor(np1)
    optimizer = torch.optim.SGD([torch0], lr=lr0)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    for _ in range(num_epoch):
        optimizer.zero_grad()
        torch.sum((torch0-torch1)**2).backward()
        optimizer.step()
        lr_scheduler.step()
    assert hfe(ret_, torch0.detach().numpy()) < 1e-7
    assert hfe(lr_current, lr_scheduler.get_last_lr()[0]) < 1e-7


def test_torch_optim_lr_scheduler_StepLR(lr0=0.233, gamma=0.95, num_epoch=5, lr_scheduler_step_size=2):
    np0 = np.random.rand(3)
    np1 = np.random.rand(3)
    ret_ = np0
    lr_current = lr0
    for ind0 in range(num_epoch):
        ret_ = ret_ - lr_current*2*(ret_-np1)
        if (ind0>0) and (ind0%lr_scheduler_step_size==(lr_scheduler_step_size-1)):
            lr_current = lr_current*gamma

    torch0 = torch.tensor(np0, requires_grad=True)
    torch1 = torch.tensor(np1)
    optimizer = torch.optim.SGD([torch0], lr=lr0)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=gamma)
    for _ in range(num_epoch):
        optimizer.zero_grad()
        torch.sum((torch0-torch1)**2).backward()
        optimizer.step()
        lr_scheduler.step()
    assert hfe(ret_, torch0.detach().numpy()) < 1e-7
    assert hfe(lr_current, lr_scheduler.get_last_lr()[0]) < 1e-7


def test_torch_optim_lr_scheduler_MultiStepLR(lr0=0.23, gamma=0.233, num_epoch=7, milestones=(1,4)):
    np0 = np.random.rand(3)
    np1 = np.random.rand(3)
    ret_ = np0
    lr_current = lr0
    tmp0 = set(milestones)
    for ind0 in range(num_epoch):
        ret_ = ret_ - lr_current*2*(ret_-np1)
        if (ind0+1) in tmp0:
            lr_current = lr_current*gamma

    torch0 = torch.tensor(np0, requires_grad=True)
    torch1 = torch.tensor(np1)
    optimizer = torch.optim.SGD([torch0], lr=lr0)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma)
    for _ in range(num_epoch):
        optimizer.zero_grad()
        torch.sum((torch0-torch1)**2).backward()
        optimizer.step()
        lr_scheduler.step()
    assert hfe(ret_, torch0.detach().numpy()) < 1e-7
    assert hfe(lr_current, lr_scheduler.get_last_lr()[0]) < 1e-7
