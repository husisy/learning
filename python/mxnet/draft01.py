import os
import numpy as np
import mxnet as mx
from collections import defaultdict
from tqdm import tqdm

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())

def show_batch_figure(batch_data, batch_label, id_to_label):
    # (mx,uint8,(25,28,28,1))
    # (mx,int32,(25,))
    import matplotlib.pyplot as plt
    plt.ion()
    fig,ax_list = plt.subplots(5, 5, figsize=(10,8))#width,height
    ax_list = [y for x in ax_list for y in x]
    for ax_i,data_i,label_i in zip(ax_list, batch_data, batch_label):
        ax_i.imshow(data_i[:,:,0].asnumpy())
        ax_i.xaxis.set_visible(False)
        ax_i.yaxis.set_visible(False)
        ax_i.set_title(id_to_label[label_i.item()])

class LeNet5(mx.gluon.nn.Block):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv0 = mx.gluon.nn.Conv2D(channels=6, kernel_size=5, activation='relu')
        self.mp0 = mx.gluon.nn.MaxPool2D(pool_size=2, strides=2)
        self.conv1 = mx.gluon.nn.Conv2D(channels=16, kernel_size=3, activation='relu')
        self.mp1 = mx.gluon.nn.MaxPool2D(pool_size=2, strides=2)
        self.fc0 = mx.gluon.nn.Dense(120, activation='relu')
        self.fc1 = mx.gluon.nn.Dense(84, activation='relu')
        self.fc2 = mx.gluon.nn.Dense(10)

    def forward(self, x):
        x = self.conv0(x)
        x = self.mp0(x)
        x = self.conv1(x)
        x = self.mp1(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def hf_lenet5():
    net = mx.gluon.nn.Sequential()
    net.add(
        mx.gluon.nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
        mx.gluon.nn.MaxPool2D(pool_size=2, strides=2),
        mx.gluon.nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
        mx.gluon.nn.MaxPool2D(pool_size=2, strides=2),
        mx.gluon.nn.Flatten(),
        mx.gluon.nn.Dense(120, activation='relu'),
        mx.gluon.nn.Dense(84, activation='relu'),
        mx.gluon.nn.Dense(10),
    )
    return net


id_to_label = dict(enumerate(['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']))

# dataset = mx.gluon.data.vision.datasets.FashionMNIST() #(list,(tuple,%,2)) %0(mx,uint8,(28,28,1)) %1(np,int32,())
# tmp0 = np.random.randint(len(dataset), size=(25,))
# show_batch_figure(*dataset[tmp0], id_to_label)

transformer = mx.gluon.data.vision.transforms.Compose([
    mx.gluon.data.vision.transforms.ToTensor(),
    mx.gluon.data.vision.transforms.Normalize(0.13, 0.31), #normal with real mean 0.13 and standard deviation 0.31
])
trainset = mx.gluon.data.vision.datasets.FashionMNIST(train=True).transform_first(transformer)
trainloader = mx.gluon.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4)

testset = mx.gluon.data.vision.FashionMNIST(train=False)
testloader = mx.gluon.data.DataLoader(testset.transform_first(transformer), batch_size=256, num_workers=4)

# net = hf_lenet5()
net = LeNet5()
net.initialize(init=mx.init.Xavier())
hf_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

metric_history = defaultdict(list)
for ind_epoch in range(5):
    with tqdm(total=len(trainloader), desc='epoch-{}'.format(ind_epoch)) as pbar:
        # net.train()
        correct = 0
        train_correct = 0
        train_total = 0
        for ind_batch, (data_i, label_i) in enumerate(trainloader):
            with mx.autograd.record():
                predict = net(data_i)
                loss = hf_loss(predict, label_i).mean()
            loss.backward()
            trainer.step(1)

            train_correct += (predict.argmax(axis=1).astype('int32')==label_i).sum().asscalar()
            train_total += label_i.shape[0]
            metric_history['train-loss'].append(loss.asscalar())
            if ind_batch+1 < len(trainloader):
                pbar.set_postfix({'loss':'{:5.3}'.format(loss.asscalar()), 'acc':'{:4.3}%'.format(100*train_correct/train_total)})
                pbar.update() #move the last update to val
        metric_history['train-acc'].append(train_correct / train_total)

        val_acc = sum((net(x).argmax(axis=1).astype('int32')==y).sum() for x,y in testloader).asscalar() / len(testset)
        metric_history['val-acc'].append(val_acc)
        pbar.set_postfix({'acc':'{:4.3}%'.format(100*train_correct/train_total), 'val-acc':'{:4.3}%'.format(val_acc*100)})
        pbar.update()


## GPU
net.save_parameters(hf_file('tbd00.mxparams'))
device = mx.gpu(0)
# net.collect_params().initialize(force_reinit=True, ctx=device)
net_gpu = LeNet5()
net_gpu.load_parameters(hf_file('tbd00.mxparams'), ctx=device)
tmp0 = ((net_gpu(x.copyto(device)).argmax(axis=1).astype('int32'),y.copyto(device)) for x,y in testloader)
val_acc = sum((x==y).sum() for x,y in tmp0).asscalar() / len(testset)

# net.save_parameters('net.params')
# TODO lr-scheduler
# TODO serialzation
