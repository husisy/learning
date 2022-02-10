import os
import PIL
import piexif
import numpy as np
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt


DATA_DIR = os.path.join('~', 'pytorch_data')
hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())

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


def _demo_basic(dataset):
    id_to_label = {y:x for x,y in dataset.class_to_idx.items()}
    # dataset[0] #(PIL,Image.Image, int)
    ind0 = np.random.randint(len(dataset), size=(25,))
    tmp0 = [np.asarray(dataset[x][0]) for x in ind0]
    tmp1 = [id_to_label[dataset[x][1]] for x in ind0]
    show_batch_figure(tmp0, tmp1)


def demo_fashion_mnist():
    dataset = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=True, download=True)
    _demo_basic(dataset)


def demo_mnist():
    dataset = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True)
    _demo_basic(dataset)


def demo_cifar10():
    dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
    _demo_basic(dataset)


def demo_ILSVRC2012():
    ILSVRC2012_DIR = '/media/hdd/pytorch_data/ILSVRC2012'
    trainset = torchvision.datasets.ImageNet(ILSVRC2012_DIR, split='train') #1281167
    validationset = torchvision.datasets.ImageNet(ILSVRC2012_DIR, split='val') #50000
    _demo_basic(trainset)

    # hf0 = lambda *x: os.path.join(ILSVRC2012_DIR, 'train', *x)
    # tmp0 = [hf0(x,y) for x in os.listdir(hf0()) for y in os.listdir(hf0(x))]
    # z0 = np.array([PIL.Image.open(x).size for x in tqdm(tmp0)]) #347 seconds on ssd
    # np.mean(z0, axis=0) # mean(train)=(473,406), mean(val)=(490,430)

    #[ERROR]too many open files #almost 2 minutes


def demo_remove_ILSVRC2012_corrupt_image():
    # see https://github.com/horovod/horovod/issues/333
    # see https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/discussion/31558
    # warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
    ILSVRC2012_DIR = '/media/hdd/pytorch_data/ILSVRC2012'
    filepath = os.path.join(ILSVRC2012_DIR, 'train/n04152593/n04152593_17460.JPEG')
    new_file = hf_file('tbd00.JPEG')
    piexif.remove(filepath, new_file)
    with PIL.Image.open(filepath) as image_pil:
        np0 = np.array(image_pil) #UserWarning: Corrupt EXIF data.  Expecting to read 4 bytes but only got 0. warnings.warn(str(msg))
    with PIL.Image.open(new_file) as image_pil:
        np1 = np.array(image_pil) #no warning
    assert np.all(np0==np1)
