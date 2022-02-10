import typing
import numpy as np
import oneflow as flow
from tqdm import tqdm

BATCH_SIZE = 100


def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(data, 32, 5, padding="SAME", activation=flow.nn.relu, name="conv1", kernel_initializer=initializer)
    pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding="SAME", name="pool1", data_format="NCHW")
    conv2 = flow.layers.conv2d(pool1, 64, 5, padding="SAME", activation=flow.nn.relu, name="conv2", kernel_initializer=initializer)
    pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding="SAME", name="pool2", data_format="NCHW")
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name="dense1")
    if train:
        hidden = flow.nn.dropout(hidden, rate=0.5, name="dropout")
    ret = flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="dense2")
    return ret


@flow.global_function(type="train")
def train_job(
        images: flow.typing.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
        labels: flow.typing.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
    ) -> typing.Tuple[flow.typing.Numpy,flow.typing.Numpy]:
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=True)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    return loss,logits


@flow.global_function(type="predict")
def eval_job(
        images: flow.typing.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    ) -> flow.typing.Numpy:
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=False)
    return logits


if __name__ == "__main__":
    flow.config.gpu_device_num(1)
    check_point = flow.train.CheckPoint()
    check_point.init()
    # check_point.load("./lenet_models_1")

    tmp0 = flow.data.load_mnist(train_batch_size=BATCH_SIZE, test_batch_size=BATCH_SIZE, out_dir='data')
    (train_images, train_labels), (test_images, test_labels) = tmp0

    for ind_epoch in range(5):
        with tqdm(total=len(train_images), desc='epoch-{}'.format(ind_epoch)) as pbar:
            num_correct_train = 0
            num_total_train = 0
            for images,labels in zip(train_images, train_labels):
                num_total_train += images.shape[0]
                loss,logits = train_job(images, labels)
                num_correct_train += np.sum(np.argmax(logits, axis=1)==labels)
                pbar.set_postfix({'loss':'{:5.3}'.format(loss.mean()), 'acc':'{:5.3}'.format(num_correct_train/num_total_train)})
                pbar.update()
    # check_point.save("./lenet_models_1")  # need remove the existed folder

    num_correct_test = 0
    num_total_test = 0
    for x,y in zip(test_images,test_labels):
        num_total_test += y.shape[0]
        num_correct_test += np.sum(np.argmax(eval_job(x), axis=1)==y)
    print('test accuracy: ', num_correct_test / num_total_test)
