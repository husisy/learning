import numpy as np
import mindspore as ms

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target='GPU')


class DummyNet00(ms.nn.Cell):
    def __init__(self, np0):
        super().__init__()
        self.ms0 = ms.Parameter(ms.Tensor(np0, dtype=ms.float32), name='ms0')
        self.op0 = ms.ops.ReduceSum()

    def construct(self, x):
        x = self.op0(self.ms0**2 * x)
        return x


def test_nn_momentum():
    # learning_rate=0.2, num_step=5, momentum=0.23, weight_decay=0.023
    learning_rate=0.2
    num_step=5
    momentum=0.23
    weight_decay=0.023
    np0 = np.random.randn(3).astype(np.float32)
    np1 = np.random.randn(3).astype(np.float32)

    np_momentum_step_i = np.zeros_like(np0)
    np0_step_i = np0
    for ind0 in range(num_step):
        np_grad_step_i = 2*np0_step_i*np1 + np0_step_i*weight_decay
        if ind0==0:
            np_momentum_step_i = np_grad_step_i
        else:
            np_momentum_step_i = momentum*np_momentum_step_i + np_grad_step_i
        np0_step_i = np0_step_i - learning_rate*np_momentum_step_i
    ret_ = np0_step_i

    net = DummyNet00(np0)
    # hf_loss = lambda x,y: x #not support yet
    def hf_loss(x,y):
        return x
    optimizer = ms.nn.Momentum(net.trainable_params(), learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay)
    train_network = ms.nn.TrainOneStepCell(ms.nn.WithLossCell(net, hf_loss), optimizer)
    for _ in range(num_step):
        loss = train_network(ms.Tensor(np1), ms.Tensor(233))
    ret0 = net.ms0.asnumpy()

    assert hfe(ret_, ret0) < 1e-5
    tmp0 = optimizer.moments[0].asnumpy()
    assert hfe(np_momentum_step_i, tmp0) < 1e-5


def test_ops_ApplyMomentum():
    shape = (2,3)
    lr = np.random.rand()
    momentum = np.random.rand()
    np_weight = np.random.randn(*shape).astype(np.float32)
    np_gradient = np.random.randn(*shape).astype(np.float32)
    np_moment = np.random.randn(*shape).astype(np.float32)
    ret_moment_ = np_moment*momentum + np_gradient
    ret_weight_ = np_weight - lr*ret_moment_

    ms_weight = ms.Parameter(ms.Tensor(np_weight))
    ms_gradient = ms.Tensor(np_gradient)
    ms_moment = ms.Parameter(ms.Tensor(np_moment))
    op = ms.ops.ApplyMomentum()
    # op = ms.ops._selected_ops.ApplyMomentum()
    op(ms_weight, ms_moment, lr, ms_gradient, momentum)
    ret_moment = ms_moment.asnumpy()
    ret_weight = ms_weight.asnumpy()

    assert hfe(ret_moment_, ret_moment) < 1e-5
    assert hfe(ret_weight_, ret_weight) < 1e-5



# TODO test custom optimizer
