import numpy as np
import chainer as ch
from tqdm import tqdm

class MyNetwork(ch.Chain):

    def __init__(self, n_mid_units=100, n_out=10):
        super().__init__()
        with self.init_scope():
            self.l1 = ch.links.Linear(None, n_mid_units)
            self.l2 = ch.links.Linear(n_mid_units, n_mid_units)
            self.l3 = ch.links.Linear(n_mid_units, n_out)

    def forward(self, x):
        h = ch.functions.relu(self.l1(x))
        h = ch.functions.relu(self.l2(h))
        return self.l3(h)


batchsize = 128
max_epoch = 3
gpu_id = 0  # CPU(-1) GPU(0/1/2)

train, test = ch.datasets.mnist.get_mnist(withlabel=True, ndim=1)
train_iter = ch.iterators.SerialIterator(train, batchsize, repeat=False, shuffle=True)
test_iter = ch.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)


model = MyNetwork()
if gpu_id >= 0:
    model.to_gpu(gpu_id)

optimizer = ch.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(model)

# TODO tqdm
loss_history = []
for ind0 in range(max_epoch):

    with tqdm(range(len(train)//batchsize+1)) as pbar:
        for ind1,(train_batch) in zip(pbar,train_iter):
            image_train = np.stack([x[0] for x in train_batch])
            target_train = np.stack([x[1] for x in train_batch])
            if gpu_id>=0:
                image_train = ch.backends.cuda.to_gpu(image_train, device=gpu_id)
                target_train = ch.backends.cuda.to_gpu(target_train, device=gpu_id)
            prediction_train = model(image_train)
            loss = ch.functions.softmax_cross_entropy(prediction_train, target_train)
            loss_history.append(float(ch.backends.cuda.to_cpu(loss.array)))
            model.cleargrads()
            loss.backward()
            optimizer.update()
            if ind1 % 10==0:
                tmp0 = '{:5.3}'.format(sum(loss_history[-10:])/10)
                pbar.set_postfix(train_loss=tmp0)
    train_iter.reset()

    test_losses = []
    test_accuracies = []
    for test_batch in test_iter:
        image_test = np.stack([x[0] for x in test_batch])
        target_test = np.stack([x[1] for x in test_batch])
        if gpu_id>=0:
            image_test = ch.backends.cuda.to_gpu(image_test, device=gpu_id)
            target_test = ch.backends.cuda.to_gpu(target_test, device=gpu_id)
        prediction_test = model(image_test)
        loss_i = ch.functions.softmax_cross_entropy(prediction_test, target_test)
        accuracy_i = ch.functions.accuracy(prediction_test, target_test)
        if gpu_id>=0:
            # loss_i = ch.backends.cuda.to_cpu(loss_i.array) #numpy.ndarray
            # accuracy_i = ch.backends.cuda.to_cpu(accuracy_i.array) #numpy.ndarray
            loss_i.to_cpu()
            accuracy_i.to_cpu()
        test_losses.append(loss_i.array)
        test_accuracies.append(accuracy_i.array)
    test_iter.reset()
    print('epoch: {}; val_loss:{:.04f} val_accuracy:{:.04f}'.format(ind0, np.mean(test_losses), np.mean(test_accuracies)))
