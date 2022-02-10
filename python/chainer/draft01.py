import os
import numpy as np

import chainer as ch

# https://raw.githubusercontent.com/chainer/chainer/master/examples/glance/mushrooms.csv
hf_data = lambda *x: os.path.join('data', *x)
assert os.path.exists(hf_data('mushrooms.csv'))

data_array = np.genfromtxt(hf_data('mushrooms.csv'), delimiter=',', dtype=str, skip_header=1)
for col in range(data_array.shape[1]):
    data_array[:, col] = np.unique(data_array[:, col], return_inverse=True)[1]
X = data_array[:, 1:].astype(np.float32)
Y = data_array[:, 0].astype(np.int32)[:, None]
train, test = ch.datasets.split_dataset_random(ch.datasets.TupleDataset(X, Y), first_size=int(data_array.shape[0]*0.7))
# chainer.datasets.sub_dataset.SubDataset

train_iter = ch.iterators.SerialIterator(train, batch_size=100)
test_iter = ch.iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)
# chainer.iterators.serial_iterator.SerialIterator

def MLP(n_units, n_out):
    layer = ch.Sequential(ch.links.Linear(n_units), ch.functions.relu)
    model = layer.repeat(2)
    model.append(ch.links.Linear(n_out))
    return model

model = ch.links.Classifier(MLP(44, 1), lossfun=ch.functions.sigmoid_cross_entropy, accfun=ch.functions.binary_accuracy)
optimizer = ch.optimizers.SGD().setup(model)
updater = ch.training.StandardUpdater(train_iter, optimizer, device=-1) #CPU(device=-1) GPU(device=0)....
trainer = ch.training.Trainer(updater, (50, 'epoch'), out='tbd00')

trainer.extend(ch.training.extensions.Evaluator(test_iter, model, device=-1))
trainer.extend(ch.training.extensions.DumpGraph('main/loss'))
trainer.extend(ch.training.extensions.snapshot(), trigger=(20, 'epoch'))
trainer.extend(ch.training.extensions.LogReport())
trainer.extend(ch.training.extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
trainer.extend(ch.training.extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))
trainer.extend(ch.training.extensions.PrintReport(['epoch', 'main/loss',
        'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

trainer.run()

x, t = test[np.random.randint(len(test))]
predict = model.predictor(x[np.newaxis]).array[0,0]
answer = 'poisonous' if predict>=0 else 'edible'
