import os
import numpy as np
import mindspore as ms
import matplotlib.pyplot as plt

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target='GPU')
# CPU is not supported in Pynative mode, ms.common.api.ms_function

MS_DATA_ROOT = os.path.expanduser('~/ms_data')

def show_batch_figure(data_list, label_list):
    plt.ion()
    fig,ax_list = plt.subplots(5, 5, figsize=(10,8))#width,height
    ax_list = [y for x in ax_list for y in x]
    for ax_i,data_i,label_i in zip(ax_list, data_list, label_list):
        ax_i.imshow(data_i)
        ax_i.xaxis.set_visible(False)
        ax_i.yaxis.set_visible(False)
        ax_i.set_title(label_i)
    fig.tight_layout()


def demo_cifar10():
    cifar10_datapath = os.path.join(MS_DATA_ROOT, 'cifar10')
    ds_train = ms.dataset.Cifar10Dataset(cifar10_datapath, 'train')
    ds_val = ms.dataset.Cifar10Dataset(cifar10_datapath, 'test')
    assert ds_train.get_dataset_size()==50000
    assert ds_val.get_dataset_size()==10000
    tmp0 = list(ds_train)
    image = [x['image'].asnumpy() for x in tmp0] #(list,(np,uint8,(32,32,3)),60000)
    label = [x['label'].asnumpy().item() for x in tmp0] #(list,int,60000)
    tmp0 = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    id_to_label = dict(enumerate(tmp0))

    ind0 = np.random.permutation(len(image))[:25]
    tmp0 = [image[x] for x in ind0]
    tmp1 = [id_to_label[label[x]] for x in ind0]
    show_batch_figure(tmp0, tmp1)


def demo_mnist():
    mnist_datapath = os.path.join(MS_DATA_ROOT, 'mnist')
    ds_train = ms.dataset.MnistDataset(os.path.join(mnist_datapath,'train'), 'train')
    ds_val = ms.dataset.MnistDataset(os.path.join(mnist_datapath,'val'), 'test')
    assert ds_train.get_dataset_size()==60000
    assert ds_val.get_dataset_size()==10000
    tmp0 = list(ds_train)
    image = [x[0].asnumpy()[:,:,0] for x in tmp0] #(list,(np,uint8,(28,28)),60000)
    label = [x[1].asnumpy().item() for x in tmp0] #(list,int,60000)
    id_to_label = {x:str(x) for x in range(10)}

    ind0 = np.random.permutation(len(image))[:25]
    tmp0 = [image[x] for x in ind0]
    tmp1 = [id_to_label[label[x]] for x in ind0]
    show_batch_figure(tmp0, tmp1)


def demo_ILSVRC2012():
    ILSVRC2012_ROOT = '/opt/pytorch_data/ILSVRC2012'
    train_dir = os.path.join(ILSVRC2012_ROOT, 'train')
    val_dir = os.path.join(ILSVRC2012_ROOT, 'val')
    id_to_label = dict(enumerate(sorted(os.listdir(train_dir))))
    label_to_id = {y:x for x,y in id_to_label.items()}
    ds_train = ms.dataset.ImageFolderDataset(train_dir, decode=True, class_indexing=label_to_id)
    ds_val = ms.dataset.ImageFolderDataset(val_dir, decode=True, class_indexing=label_to_id)
    assert ds_train.get_dataset_size()==1281167
    assert ds_val.get_dataset_size()==50000

    tmp0 = ds_train.shuffle(1000)
    z0 = [x for _,x in zip(range(25), iter(tmp0))]
    image = [x[0].asnumpy() for x in z0]
    label = [id_to_label[x[1].asnumpy().item()] for x in z0]
    show_batch_figure(image, label)
