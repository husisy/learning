import os
import numpy as np
import mindspore as ms
import mindspore.dataset.transforms

from utils import next_tbd_dir

ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target='GPU')
hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

class DummyNet00(ms.nn.Cell):
    def __init__(self, N0=5):
        super(DummyNet00, self).__init__()
        self.fc0 = ms.nn.Dense(N0, 23)
        self.fc1 = ms.nn.Dense(23, 1)
        self.relu = ms.nn.ReLU()

    def construct(self, x):
        x = self.fc0(x)
        x = self.relu(x)
        x = self.fc1(x)[:,0]
        return x


def test_cell_checkpoint():
    N0 = 233
    N1 = 5
    logdir = next_tbd_dir()
    np0 = np.random.randn(N0, N1).astype(np.float32)
    ms0 = ms.Tensor(np0)
    net = DummyNet00(N1)
    model = ms.train.Model(net)
    ret_ = model.predict(ms0)

    ckpt_path = os.path.join(logdir, '2333.ckpt')
    ms.train.serialization.save_checkpoint(net, ckpt_path)

    net1 = DummyNet00(N1)
    model1 = ms.train.Model(net1)
    tmp0 = ms.train.serialization.load_checkpoint(ckpt_path)
    ms.train.serialization.load_param_into_net(net1, tmp0)
    ret0 = model1.predict(ms0)
    assert hfe(ret_.asnumpy(), ret0.asnumpy()) < 1e-5


def test_train_checkpoint():
    N0 = 233
    N1 = 5
    np0 = np.random.randn(N0, N1).astype(np.float32)
    np1 = np.random.randn(N0,).astype(np.float32)

    # if drop_remainder=False, some strange error will raise below
    ds0 = ms.dataset.NumpySlicesDataset((np0,np1), column_names=['x','y'], shuffle=False).batch(4, drop_remainder=True)
    hf_loss = ms.nn.loss.MSELoss(reduction='mean')

    logdir = next_tbd_dir()
    ms0 = ms.Tensor(np0)
    net = DummyNet00(N1)
    optimizer = ms.nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
    model = ms.train.Model(net, hf_loss, optimizer, metrics={'mse': ms.nn.metrics.MSE()})
    model.train(1, ds0)
    ret_ = model.predict(ms.Tensor(np0))

    ckpt_path = os.path.join(logdir, '2333.ckpt')
    ms.train.serialization.save_checkpoint(optimizer, ckpt_path)

    z0 = dict(list(net.parameters_and_names()))
    z1 = dict(list(optimizer.parameters_and_names()))
    assert set(z0.keys()) <= set(z1.keys())
    assert all(id(v)==id(z1[k]) for k,v in z0.items()) #all tensor of net are included in optimzer
    z2 = ms.train.serialization.load_checkpoint(ckpt_path)
    assert set(z1.keys())==set(z2.keys())
    assert all(hfe(v.asnumpy(), z2[k].asnumpy())<1e-5 for k,v in z1.items())

    net1 = DummyNet00(N1)
    optimizer1 = ms.nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
    model1 = ms.train.Model(net1, hf_loss, optimizer1, metrics={'mse': ms.nn.metrics.MSE()})
    tmp0 = ms.train.serialization.load_checkpoint(ckpt_path)
    ms.train.serialization.load_param_into_net(optimizer1, tmp0)
    # ms.train.serialization.load_param_into_net(net1, tmp0) #no need
    ret0 = model.predict(ms.Tensor(np0))
    assert hfe(ret_.asnumpy(), ret0.asnumpy()) < 1e-5
