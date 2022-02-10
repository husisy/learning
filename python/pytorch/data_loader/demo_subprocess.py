import os
import math
import time
import numpy as np
import torch

class MyDataset00(torch.utils.data.Dataset):
    def __init__(self, num_sample):
        self.num_sample = num_sample
        self.start_time = time.time()

    def __getitem__(self, index):
        tmp0 = time.time() - self.start_time
        print(f'[MyDataset00.__getitem__()] index={index}, pid={os.getpid()}, time={tmp0:.3}') #in subprocess
        time.sleep(0.1)
        return index
    def __len__(self):
        return self.num_sample


def my_collate(batch_data):
    # in subprocess
    print(f'[my_collate] pid={os.getpid()}') #in different process
    ret = torch.from_numpy(np.stack(batch_data))
    return ret


# __getitem__() and collate_fn() both are running in the subprocess
# prefetch is used
def demo_subprocess_pid():
    print('# demo_subprocess_pid')
    print(f'[main] pid={os.getpid()}')
    num_batch = 5
    batch_size = 3
    dataset = MyDataset00(num_batch*batch_size)
    start_time = dataset.start_time
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=2, collate_fn=my_collate)
    z0 = iter(dataloader)
    for ind0 in range(len(dataloader)):
        tmp0 = time.time() - start_time
        print(f'[main-for-loop] step={ind0} start at {tmp0:.3}')
        _ = next(z0)
        tmp0 = time.time() - start_time
        print(f'[main-for-loop] step={ind0} receive data at {tmp0:.3}')
        time.sleep(0.3)


# TODO dataset to match sampler
class MyTrainSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, num_sample, batch_size, seed=None, world_size=1, rank=0):
        num_batch = math.ceil(num_sample/(world_size*batch_size))
        num_padding = num_batch*world_size*batch_size - num_sample
        rng = np.random.RandomState(seed)
        self.index_base = np.arange(num_sample)
        self.num_batch = num_batch
        self.num_sample = num_sample
        self.num_padding = num_padding
        self.batch_size = batch_size
        self.world_size = world_size
        self.rng = rng
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.index = None
    def _new_index(self):
        self.index = np.concatenate([self.index_base, self.rng.choice(self.index_base, size=self.num_padding)]) #replace=True
        self.rng.shuffle(self.index)
        self.index = self.index.reshape(self.world_size, self.num_batch*self.batch_size)[self.rank]
    def __len__(self):
        return self.num_sample + self.num_padding
    def set_epoch(self, epoch):
        raise Exception('do not use this method, use __init__(seed=xxx) instead')
    def __iter__(self):
        self._new_index()
        yield from self.index


class MyDataset01(torch.utils.data.Dataset):
    def __init__(self, num_sample):
        self.num_sample = num_sample
    def __getitem__(self, index):
        ret = np.array([os.getpid(), index])
        return ret
    def __len__(self):
        return self.num_sample


def demo_subprocess_sampler():
    print('# demo_subprocess_sampler')
    num_sample = 13
    batch_size = 3
    num_workers = 3
    seed = 233
    num_epoch = 3

    sampler = MyTrainSampler(num_sample, batch_size, seed=seed)
    for ind0 in range(num_epoch):
        next(iter(sampler))
        print(f'[epoch={ind0}] MyTrainSampler.index:', sampler.index)

    sampler01 = MyTrainSampler(num_sample, batch_size, seed=seed)
    dataset = MyDataset01(num_sample)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, sampler=sampler01, num_workers=num_workers)
    for ind0 in range(num_epoch):
        # every iter will create new subprocess
        print(f'[epoch={ind0}]')
        ret = torch.stack(list(iter(dataloader))).reshape(-1, batch_size*2)
        print(ret)


# TODO dataset to match sampler

class MyPrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, sampler, num_workers):
        self.num_sample = num_sample
        self.sampler = sampler
        self.ind_next_data = None
    def __getitem__(self, index):
        if self.ind_next_data is None:
            next(iter(self.sampler))
        ret = np.array([os.getpid(), index])
        return ret
    def __len__(self):
        return self.num_sample
    def guess_process(self, index):
        pass


if __name__=='__main__':
    demo_subprocess_pid()
    # print()
    # demo_subprocess_sampler()
