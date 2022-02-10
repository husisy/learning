import os
import numpy as np
import mindspore as ms
import mindspore.dataset.transforms

ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target='GPU')

def create_mnist_dataset(data_root, phase, batch_size=128):
    assert phase in {'train','val'}
    image_op = [
        ms.dataset.vision.c_transforms.Resize((32, 32), interpolation=ms.dataset.vision.Inter.LINEAR),
        ms.dataset.vision.c_transforms.Rescale(1/255, 0),
        ms.dataset.vision.c_transforms.Rescale(3.2457, -0.4242),
        ms.dataset.vision.c_transforms.HWC2CHW(),
    ]
    label_op = [ms.dataset.transforms.c_transforms.TypeCast(ms.int32)]

    if phase=='train':
        ds = ms.dataset.MnistDataset(os.path.join(data_root, 'train'), shuffle=False)
    else:
        ds = ms.dataset.MnistDataset(os.path.join(data_root, 'val'), shuffle=False)
    ds = ds.map(operations=label_op, input_columns="label", num_parallel_workers=1)
    ds = ds.map(operations=image_op, input_columns="image", num_parallel_workers=1)
    if phase=='train':
        ds = ds.shuffle(buffer_size=10000)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


class AccMonitor(ms.train.callback.Callback):
    def __init__(self, model, ds_eval):
        super(AccMonitor, self).__init__()
        self.model = model
        self.ds_val = ds_val
        self.len_dataset = ds_val.get_dataset_size() * ds_val.get_batch_size() #TODO bad

    def epoch_end(self, run_context):
        tmp0 = self.model.eval(self.ds_val)['top1']
        print(f'[validation] acc1={tmp0}')


class LeNet5(ms.nn.Cell):
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        Normal = ms.common.initializer.Normal
        self.conv1 = ms.nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = ms.nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = ms.nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = ms.nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = ms.nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = ms.nn.ReLU()
        self.max_pool2d = ms.nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = ms.nn.Flatten()

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# TODO test accuracy error when batchsize differs
x_train_np = np.random.randn(2048, 3).astype(np.float32)
y_train_np = np.random.randint(0, 10, size=(2048,))
x_test_np = np.random.randn(1024, 3).astype(np.float32)
y_test_np = np.random.randint(0, 10, size=(256,))

ds_train = ms.dataset.NumpySlicesDataset((x_train_np,y_train_np), column_names=['x','y'], shuffle=False).shuffle(buffer)

ds_train = ms.dataset.NumpySlicesDataset((np0,), column_names=['np0'], shuffle=False)

mnist_root = os.path.expanduser('~/ms_data/mnist')
ds_train = create_mnist_dataset(mnist_root, 'train', 128)
ds_val = create_mnist_dataset(mnist_root, 'val', 128)

net_loss = ms.nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
net = LeNet5()
optimizer = ms.nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
model = ms.train.Model(net, net_loss, optimizer, metrics={'top1':ms.nn.metrics.Accuracy()})

callback_list = [
    ms.train.callback.LossMonitor(per_print_times=100),
    AccMonitor(model, ds_val)
]
model.train(2, ds_train, callbacks=callback_list, dataset_sink_mode=True)

acc = model.eval(ds_val, dataset_sink_mode=True)
print('accuracy:', acc)

ds_val256 = create_mnist_dataset(mnist_root, 'val', 256)
print('ds_val256:', model.eval(ds_val256, dataset_sink_mode=True))
