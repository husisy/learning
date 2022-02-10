import numpy as np
import mindspore as ms

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="GPU")


def test_L1Loss():
    hf_loss = ms.nn.L1Loss()
    # hf_loss could be normal python function, but cannot be lambda function
    np0 = np.random.rand(5,23).astype(np.float32)
    np1 = np.random.rand(5,23).astype(np.float32)
    ret0 = hf_loss(ms.Tensor(np0), ms.Tensor(np1)).asnumpy()
    ret_ = np.abs(np0-np1).mean()
    assert hfe(ret_,ret0) < 1e-5


def test_conv():
    N, C, H, W = 1, 1, 8, 8
    Cout, Cin, Hk, Wk = 2, 1, 3, 3
    npx = np.random.rand(N,C,H,W).astype(np.float32)
    npw = np.random.rand(Cout,Cin,Hk,Wk).astype(np.float32)

    conv2d = ms.ops.Conv2D(out_channel=Cout, kernel_size=Hk, pad_mode='valid')
    ret0 = conv2d(ms.Tensor(npx), ms.Tensor(npw)).asnumpy()
    conv2d = ms.nn.Conv2d(in_channels=Cin, out_channels=Cout, kernel_size=Hk, pad_mode='valid')
    conv2d.weight.set_data(ms.Tensor(npw))
    ret1 = conv2d(ms.Tensor(npx)).asnumpy()
    assert hfe(ret0, ret1) < 1e-5


def test_batchnorm2d_train(num_batch=3, N0=3, N1=5, N2=7, N3=11, momentum=0.233, eps=0.00233):
    np_weight = np.random.randn(N1).astype(np.float32)
    np_bias = np.random.randn(N1).astype(np.float32)
    batch_np_list = [np.random.randn(N0,N1,N2,N3).astype(np.float32) for _ in range(num_batch)]

    running_mean_ = 0
    running_var_ = 1
    for batch_i in batch_np_list:
        running_mean_ = momentum*running_mean_ + (1-momentum)*batch_i.mean(axis=(0,2,3))
        running_var_ = momentum*running_var_ + (1-momentum)*batch_i.var(axis=(0,2,3), ddof=1)
    tmp0 = batch_np_list[-1]
    tmp1 = (tmp0 - tmp0.mean(axis=(0,2,3),keepdims=True)) / np.sqrt(tmp0.var(axis=(0,2,3),keepdims=True) + eps)
    ret_ = tmp1*np_weight[:,np.newaxis,np.newaxis] + np_bias[:,np.newaxis,np.newaxis]


    ms_weight = ms.Tensor(np_weight)
    ms_bias = ms.Tensor(np_bias)
    batch_ms_list = [ms.Tensor(x) for x in batch_np_list]
    bn0 = ms.nn.BatchNorm2d(N1, eps=eps, momentum=momentum, gamma_init=ms_weight, beta_init=ms_bias)
    bn0.set_train(True)
    for batch_i in batch_ms_list:
        ret0 = bn0(batch_i).asnumpy()

    bn_params = {k:v.asnumpy() for k,v in bn0.parameters_dict().items()}
    assert hfe(running_mean_, bn_params['mean']) < 1e-4
    assert hfe(running_var_, bn_params['variance']) < 1e-4
    assert hfe(np_weight, bn_params['gamma']) < 1e-6
    assert hfe(np_bias, bn_params['beta']) < 1e-6
    assert hfe(ret_, ret0) < 1e-4


def test_batchnorm2d_eval(N0=3, N1=5, N2=7, N3=11, momentum=0.233, eps=0.00233):
    np_weight = np.random.randn(N1).astype(np.float32)
    np_bias = np.random.randn(N1).astype(np.float32)
    np0 = np.random.randn(N0,N1,N2,N3).astype(np.float32)
    np_running_mean = np.random.randn(N1).astype(np.float32)
    np_running_var = np.random.uniform(1, 2, size=(N1,)).astype(np.float32)

    tmp0 = (np0 - np_running_mean[:,np.newaxis,np.newaxis]) / np.sqrt(np_running_var[:,np.newaxis,np.newaxis] + eps)
    ret_ = tmp0*np_weight[:,np.newaxis,np.newaxis] + np_bias[:,np.newaxis,np.newaxis]

    ms_weight = ms.Tensor(np_weight)
    ms_bias = ms.Tensor(np_bias)
    ms_running_mean = ms.Tensor(np_running_mean)
    ms_running_var = ms.Tensor(np_running_var)

    bn0 = ms.nn.BatchNorm2d(N1, eps=eps, momentum=momentum, gamma_init=ms_weight, beta_init=ms_bias,
            moving_mean_init=ms_running_mean, moving_var_init=ms_running_var)
    bn0.set_train(False)
    ret0 = bn0(ms.Tensor(np0)).asnumpy()

    bn_params = {k:v.asnumpy() for k,v in bn0.parameters_dict().items()}
    assert hfe(np_running_mean, bn_params['mean']) < 1e-4
    assert hfe(np_running_var, bn_params['variance']) < 1e-4
    assert hfe(np_weight, bn_params['gamma']) < 1e-6
    assert hfe(np_bias, bn_params['beta']) < 1e-6
    assert hfe(ret_, ret0) < 1e-4
