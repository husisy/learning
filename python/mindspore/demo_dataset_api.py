import numpy as np
import mindspore as ms

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target='GPU')
# CPU is not supported in Pynative mode, ms.common.api.ms_function



def demo_numpy_dataset():
    N0 = 23
    np0 = np.random.randn(N0,3)
    np1 = np.random.randn(N0)
    ds0 = ms.dataset.NumpySlicesDataset((np0,np1), column_names=['np0','np1'], shuffle=False)
    ds0.output_shapes() #[[3], []]
    ds0.output_types() #[np.float64,np.float64]
    tmp0 = list(ds0)
    assert np.all(np0==np.stack([x[0].asnumpy() for x in tmp0]))
    assert np.all(np1==np.stack([x[1].asnumpy() for x in tmp0]))


def demo_dataset_map():
    np0 = np.random.rand(11, 3)
    hf0 = lambda x: 0.233*x #should

    ds0 = ms.dataset.NumpySlicesDataset((np0,), column_names=['np0'], shuffle=False)
    ds1 = ds0.map(operations=hf0, input_columns=['np0'])
    ret_ = hf0(np0)
    ret0 = np.stack([x[0].asnumpy() for x in ds1])
    assert np.all(ret_==ret0)


def demo_dataset_batch():
    np0 = np.random.rand(11)
    batch_size = 2

    ds0 = ms.dataset.NumpySlicesDataset((np0,), column_names=['np0'], shuffle=False)
    ds1 = ds0.batch(batch_size, drop_remainder=False)
    ret0 = np.concatenate([x[0].asnumpy() for x in ds1])
    assert np.all(ret0==np0)

    ds0 = ms.dataset.NumpySlicesDataset((np0,), column_names=['np0'], shuffle=False)
    ds1 = ds0.batch(batch_size, drop_remainder=True)
    ret_ = np0[:((len(np0)//batch_size)*batch_size)].reshape(-1, batch_size)
    assert np.all(np.stack([x[0].asnumpy() for x in ds1])==ret_)


def demo_dataset_batch_repeat_order():
    np0 = np.random.rand(5)
    batch_size = 2

    ds0 = ms.dataset.NumpySlicesDataset((np0,), column_names=['np0'], shuffle=False)
    ds1 = ds0.batch(batch_size, drop_remainder=False).repeat(2)
    tmp0 = [x[0].asnumpy() for x in ds1] #some batch may contains elements less then batch_size
    assert (len(tmp0)%2)==0
    ret0 = np.concatenate(tmp0[:(len(tmp0)//2)])
    assert np.all(np0==ret0)
    ret0 = np.concatenate(tmp0[(len(tmp0)//2):])
    assert np.all(np0==ret0)

    ds0 = ms.dataset.NumpySlicesDataset((np0,), column_names=['np0'], shuffle=False)
    ds1 = ds0.repeat(2).batch(batch_size, drop_remainder=False)
    tmp0 = [x[0].asnumpy() for x in ds1] #all batchs excluding last one contain exactly batch_size items
    assert all(len(x)==batch_size for x in tmp0[:-1])
    ret_ = np.concatenate([np0,np0])
    ret0 = np.concatenate(tmp0)
    assert np.all(ret_==ret0)


def demo_shuffle():
    N0 = 100
    num_repeat = 5
    np0 = np.arange(N0)

    ds0 = ms.dataset.NumpySlicesDataset((np0,), column_names=['np0'], shuffle=False)
    ds1 = ds0.shuffle(buffer_size=N0)
    z0 = [np.stack([x[0].asnumpy() for x in ds1]) for _ in range(num_repeat)]
    # all repeats are differ
    assert all((not np.all(z0[x]==z0[y])) for x in range(num_repeat-1) for y in range(x+1,num_repeat))


def demo_dataset_shuffle_repeat():
    N0 = 23
    num_repeat = 5
    np0 = np.arange(N0)

    ds0 = ms.dataset.NumpySlicesDataset((np0,), column_names=['np0'], shuffle=False)
    ds1 = ds0.shuffle(buffer_size=N0).repeat(num_repeat)
    z0 = np.stack([x[0].asnumpy() for x in ds1]).reshape(num_repeat,-1)
    # data belonging to different epoch is not mixed
    assert all(np.all(np.sort(x)==np0) for x in z0)
    # all repeats are diff
    assert all((not np.all(z0[x]==z0[y])) for x in range(num_repeat-1) for y in range(x+1,num_repeat))


class MyDataset:
    def __init__(self, npx, npy):
        assert npx.shape[0]==npy.shape[0]
        self.npx = npx
        self.npy = npy
        self.len_dataset = npx.shape[0]
    def __getitem__(self, index):
        ret = self.npx[index], self.npy[index]
        return ret
    def __len__(self):
        return self.len_dataset

def demo_custom_dataset():
    N0 = 23
    npx = np.random.randn(N0, 3).astype(np.float32)
    npy = np.random.randn(N0).astype(np.float32)
    my_ds = MyDataset(npx, npy)
    ds0 = ms.dataset.GeneratorDataset(my_ds, ['npx','npy'], shuffle=False)

# TODO zip
# TODO concat


# TODO ms.dataset.vision.c_transforms


def demo_sampler():
    # TODO sampler
    # sampler0 = ms.dataset.SequentialSampler(num_samples=5)
    sampler0 = ms.dataset.RandomSampler(num_samples=5)
