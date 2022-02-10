import numpy as np
import oneflow as flow
from tqdm import tqdm

BATCH_SIZE = 100


@flow.global_function(type="train")
def train_job(
    images: flow.typing.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: flow.typing.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> flow.typing.Numpy:
    with flow.scope.placement("cpu", "0:0"):
        initializer = flow.truncated_normal(0.1)
        reshape = flow.reshape(images, [images.shape[0], -1])
        hidden = flow.layers.dense(
            reshape,
            100,
            activation=flow.nn.relu,
            kernel_initializer=initializer,
            name="dense1",
        )
        logits = flow.layers.dense(
            hidden, 10, kernel_initializer=initializer, name="dense2"
        )
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)

    return loss


if __name__ == "__main__":
    check_point = flow.train.CheckPoint()
    check_point.init()

    tmp0 = flow.data.load_mnist(train_batch_size=BATCH_SIZE, test_batch_size=BATCH_SIZE, out_dir='data')
    (train_images, train_labels), (test_images, test_labels) = tmp0
    for ind_epoch in range(2):
        with tqdm(total=len(train_images), desc='epoch-{}'.format(ind_epoch)) as pbar:
            for images,labels in zip(train_images, train_labels):
                loss = train_job(images, labels)
                pbar.set_postfix({'loss':'{:5.3}'.format(loss.mean())})
                pbar.update()

