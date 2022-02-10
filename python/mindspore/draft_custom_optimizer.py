import numpy as np
import mindspore as ms

ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target='GPU')

_momentum_opt = ms.ops.MultitypeFuncGraph("momentum_opt")


@_momentum_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _tensor_run_opt_ext(opt, momentum, learning_rate, gradient, weight, moment):
    """Apply momentum optimizer to the weight parameter using Tensor."""
    success = ms.ops.depend(True, opt(weight, moment, learning_rate, gradient, momentum))
    return success

class MyMomentum(ms.nn.optim.Optimizer):
    def __init__(self, params, learning_rate, momentum, weight_decay=0.0, loss_scale=1.0, use_nesterov=False):
        super().__init__(learning_rate, params, weight_decay, loss_scale)
        if isinstance(momentum, float) and momentum < 0.0:
            raise ValueError("momentum should be at least 0.0, but got momentum {}".format(momentum))
        self.momentum = ms.Parameter(ms.Tensor(momentum, ms.float32), name="momentum")
        self.params = self.parameters
        self.moments = self.params.clone(prefix="moments", init='zeros')
        self.hyper_map = ms.ops.HyperMap()
        self.opt = ms.ops._selected_ops.ApplyMomentum(use_nesterov=self.use_nesterov)
        assert (not self.is_group) and (not self.is_group_lr), 'group parameter is tedious, maybe later'
        assert all((not x) for x in self.ps_parameters), 'parameter_server shi za fei shi ya'

    def construct(self, gradients):
        gradients = self.decay_weight(gradients)
        gradients = self.scale_grad(gradients)
        tmp0 = ms.ops.partial(_momentum_opt, self.opt, self.momentum, self.get_lr())
        success = self.hyper_map(tmp0, gradients, self.params, self.moments)
        return success


class DummyNet01(ms.nn.Cell):
    def __init__(self, num0):
        super().__init__()
        self.fc0 = ms.nn.Dense(num0, 5)
        self.relu = ms.ops.ReLU()
        self.fc1 = ms.nn.Dense(5, 1)

    def construct(self, x):
        x = self.fc0(x)
        x = self.relu(x)
        x = self.fc1(x)[:,0]
        return x

data_np = np.random.randn(1024, 3).astype(np.float32)
label_np = np.random.randn(1024).astype(np.float32)
data_ms = ms.Tensor(data_np, dtype=ms.float32)
label_ms = ms.Tensor(label_np, dtype=ms.float32)

ds_train = ms.dataset.NumpySlicesDataset((data_np,label_np), column_names=['np0','np1'], shuffle=True).batch(32)
net = DummyNet01(data_np.shape[1])
hf_loss = ms.nn.loss.MSELoss()
optimizer = MyMomentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)
model = ms.Model(net, hf_loss, optimizer)
model.train(3, ds_train, callbacks=ms.train.callback.LossMonitor(), dataset_sink_mode=True)

# you can do some check in PYNATIVE mode
