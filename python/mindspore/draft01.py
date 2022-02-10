import numpy as np
import mindspore as ms

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target='GPU')
# CPU is not supported in Pynative mode, ms.common.api.ms_function


class DummyNet00(ms.nn.Cell):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super().__init__()
        self.conv2d = ms.ops.Conv2D(out_channel, kernel_size)
        self.bias_add = ms.ops.BiasAdd()
        tmp0 = ms.common.initializer.initializer('normal', [out_channel, in_channel, kernel_size, kernel_size])
        self.weight = ms.Parameter(tmp0, name='conv.weight')
        tmp0 = ms.common.initializer.initializer('normal', [out_channel])
        self.bias = ms.Parameter(tmp0, name='conv.bias')

    def construct(self, x):
        x = self.conv2d(x, self.weight)
        x = self.bias_add(x, self.bias)
        return x

in_channel = 5
out_channel = 7
net = DummyNet00(in_channel, out_channel)
net.trainable_params()
net.parameters_dict() #{'conv.weight': (7, 5, 3, 3), 'conv.bias': (7,)}
net(ms.Tensor(np.random.rand(5, in_channel, 224, 224).astype(np.float32)))
net.requires_grad #False(default)
net.set_grad() #seems useless
net.requires_grad #True


class LeNet(ms.nn.Cell):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = ms.ops.ReLU()
        init = ms.common.initializer.HeNormal()

        self.conv1 = ms.nn.Conv2d(1, 6, kernel_size=5, pad_mode='valid', weight_init=init)
        self.conv2 = ms.nn.Conv2d(6, 16, kernel_size=5, pad_mode='valid', weight_init=init)
        self.pool = ms.nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = ms.ops.Reshape()
        self.fc1 = ms.nn.Dense(400, 120, weight_init=init)
        self.fc2 = ms.nn.Dense(120, 84, weight_init=init)
        self.fc3 = ms.nn.Dense(84, 10, weight_init=init)

    def construct(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.reshape(x, (x.shape[0], -1))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

N0 = 32
data = ms.Tensor(np.random.randn(N0, 1, 32, 32).astype(np.float32)*0.01)
label = ms.Tensor(np.random.randint(0, 10, size=(N0,), dtype=np.int32))
net = LeNet()

optimizer = ms.nn.optim.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
hf_loss = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
net_with_criterion = ms.nn.WithLossCell(net, hf_loss)
train_network = ms.nn.TrainOneStepCell(net_with_criterion, optimizer)  # optimizer
train_network.set_train()
for ind_batch in range(5):
    loss = train_network(data, label)


# ms.nn.WithLossCell
# ms.nn.TrainOneStepCell


class DummyNet01(ms.nn.Cell):
    def __init__(self, num0):
        super().__init__()
        self.fc0 = ms.nn.Dense(num0, 5)
        self.relu = ms.ops.ReLU()
        self.fc1 = ms.nn.Dense(5, 1)

    def construct(self, x):
        # x = self.fc0(x) + np.array(2.33) #fail
        x = self.fc0(x) + 2.33 #pass
        x = self.relu(x)
        x = self.fc1(x)[:,0]
        return x

data_np = np.random.randn(1024, 3).astype(np.float32)
label_np = np.random.randn(1024).astype(np.float32)
data_ms = ms.Tensor(data_np, dtype=ms.float32)
label_ms = ms.Tensor(label_np, dtype=ms.float32)

net = DummyNet01(data_np.shape[1])
hf_loss = ms.nn.loss.MSELoss()
optimizer = ms.nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
train_network = ms.nn.TrainOneStepCell(ms.nn.WithLossCell(net, hf_loss), optimizer)
train_network.set_train()
for _ in range(10):
    loss = train_network(data_ms, label_ms)

ds_train = ms.dataset.NumpySlicesDataset((data_np,label_np), column_names=['np0','np1'], shuffle=True).batch(32)
net = DummyNet01(data_np.shape[1])
hf_loss = ms.nn.loss.MSELoss()
optimizer = ms.nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)
model = ms.Model(net, hf_loss, optimizer)
model.train(3, ds_train, callbacks=ms.train.callback.LossMonitor(), dataset_sink_mode=True)


ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target='GPU')
class DummyNet02(ms.nn.Cell):
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


class GradWrap(ms.nn.Cell):
    def __init__(self, net):
        super().__init__(auto_prefix=False)
        self.net = net
        self.weights = ms.ParameterTuple(net.trainable_params())

    def construct(self, x, label):
        assert ms.context.get_context('mode')==ms.context.PYNATIVE_MODE
        return ms.ops.GradOperation(get_by_list=True)(self.net, self.weights)(x, label)

data_np = np.random.randn(1024, 3).astype(np.float32)
label_np = np.random.randn(1024).astype(np.float32)
ds_train = ms.dataset.NumpySlicesDataset((data_np,label_np), column_names=['np0','np1'], shuffle=True).batch(32)

net = DummyNet02(data_np.shape[1])
optimizer = ms.nn.Momentum(net.trainable_params(), 0.1, momentum=0.9)
hf_loss = ms.nn.loss.MSELoss()
train_network = GradWrap(ms.nn.WithLossCell(net, hf_loss))
train_network.set_train()

data_ms,label_ms = next(iter(ds_train))
grads = train_network(data_ms, label_ms) #(list,Tensor)
optimizer(grads)
