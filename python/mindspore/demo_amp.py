import numpy as np
import mindspore as ms

ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target='GPU')

class DummyNet00(ms.nn.Cell):
    def __init__(self, input_channel):
        super().__init__()
        self.dense = ms.nn.Dense(input_channel, 1)
        self.relu = ms.ops.ReLU()

    def construct(self, x):
        x = self.dense(x)[:,0]
        x = self.relu(x)
        return x

data_ms = ms.Tensor(np.random.randn(64,23).astype(np.float32)*0.01, dtype=ms.float32)
label_ms = ms.Tensor(np.random.randn(64).astype(np.float32), dtype=ms.float32)
net = DummyNet00(data_ms.shape[1])
hf_loss = ms.nn.loss.MSELoss()
optimizer = ms.nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
model = ms.amp.build_train_network(net, optimizer, hf_loss, level="O3", loss_scale_manager=None)
loss = model(data_ms, label_ms) #ms.float32
# net.dense(data_ms) #ms.float16


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

N0 = 1024
data_np = np.random.rand(N0, 3).astype(np.float32)
label_np = np.random.rand(N0).astype(np.float32)
ds_train = ms.dataset.NumpySlicesDataset((data_np,label_np), column_names=['data','label'], shuffle=False).batch(32)

net = DummyNet01(data_np.shape[1])
hf_loss = ms.nn.loss.MSELoss()
optimizer = ms.nn.optim.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
model = ms.train.Model(net, loss_fn=hf_loss, optimizer=optimizer, amp_level="O3")
net.fc0(next(iter(ds_train))[0]) #float16, if not amp_level, then dtype=float32
model.train(epoch=10, train_dataset=ds_train)
