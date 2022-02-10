import numpy as np
import chainer as ch

class MLP(ch.Chain):

    def __init__(self, n_mid_units=100, n_out=10):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = ch.links.Linear(None, n_mid_units)
            self.l2 = ch.links.Linear(None, n_mid_units)
            self.l3 = ch.links.Linear(None, n_out)

    def forward(self, x):
        h1 = ch.functions.relu(self.l1(x))
        h2 = ch.functions.relu(self.l2(h1))
        return self.l3(h2)


gpu_id = -1 #CPU(-1) GPU(0/1/2/3)
max_epoch = 3
batchsize = 128

train, test = ch.datasets.mnist.get_mnist()
train_iter = ch.iterators.SerialIterator(train, batchsize)
test_iter = ch.iterators.SerialIterator(test, batchsize, False, False)

model = MLP()
if gpu_id >= 0:
    model.to_gpu(gpu_id)


# Wrap your model by Classifier and include the process of loss calculation within your model.
# Since we do not specify a loss function here, the default 'softmax_cross_entropy' is used.
model = ch.links.Classifier(model)

optimizer = ch.optimizers.MomentumSGD()
optimizer.setup(model)
updater = ch.training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)
trainer = ch.training.Trainer(updater, (max_epoch, 'epoch'), out='tbd00')

trainer.extend(ch.training.extensions.LogReport())
trainer.extend(ch.training.extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
trainer.extend(ch.training.extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}'))
trainer.extend(ch.training.extensions.Evaluator(test_iter, model, device=gpu_id))
trainer.extend(ch.training.extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
trainer.extend(ch.training.extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
trainer.extend(ch.training.extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
trainer.extend(ch.training.extensions.DumpGraph('main/loss'))

trainer.run()
# dot -Tpng tbd00/cg.dot -o tbd00/cg.png
