import numpy as np
import torch


def demo_dataloader_drop_last():
    # shuffle then drop-last
    np0 = np.arange(9)
    ds0 = torch.utils.data.TensorDataset(torch.tensor(np0))
    dataloader0 = torch.utils.data.DataLoader(ds0, batch_size=4, shuffle=True, drop_last=True)
    for _ in range(5):
        print(np.sort(np.concatenate([x[0].numpy() for x in dataloader0])))


def demo_DistributedSampler():
    np0 = np.arange(10) + 1
    dataset0 = torch.utils.data.TensorDataset(torch.tensor(np0))
    num_replicas = 4

    z0 = []
    random_seed = np.random.randint(1000)
    for ind0 in range(num_replicas):
        sampler = torch.utils.data.distributed.DistributedSampler(dataset0, num_replicas=num_replicas, rank=ind0, shuffle=False)
        train_loader = torch.utils.data.DataLoader(dataset0, batch_size=2, shuffle=False, sampler=sampler)
        sampler.set_epoch(random_seed)
        z0.append([tuple(x[0].numpy().tolist()) for x in train_loader])
    for ind0,x in enumerate(z0):
        print(f'[rank={ind0}]', x)
    assert set(np0.tolist()) == set(z for x in z0 for y in x for z in y) #all replicas make the whole dataset

    # for one replica, train_loader is fully controlled by the set_epoch(seed)
    random_seed = 233
    sampler = torch.utils.data.distributed.DistributedSampler(dataset0, num_replicas=num_replicas, rank=1, shuffle=True)
    train_loader = torch.utils.data.DataLoader(dataset0, batch_size=2, shuffle=False, sampler=sampler)
    sampler.set_epoch(random_seed)
    ret_ = np.concatenate([x[0].numpy() for x in train_loader])
    print('ret_:', ret_)
    for _ in range(3):
        tmp0 = np.concatenate([x[0].numpy() for x in train_loader])
        print(f'tmp0[random_seed={random_seed}]:', tmp0)
        assert np.all(ret_==tmp0) #if not set_epoch, the output is exactly the same
    for random_seed in [234,235,236]:
        sampler.set_epoch(random_seed)
        tmp0 = np.concatenate([x[0].numpy() for x in train_loader])
        print(f'tmp0[random_seed={random_seed}]:', tmp0)
        assert not np.all(ret_==tmp0)


# TODO DistributedSampler is wrong when validating datasets, always append the first several shuffled data
def demo_DistributedSampler_set_epoch():
    # see https://github.com/pytorch/pytorch/issues/25162
    # see https://github.com/pytorch/pytorch/pull/32951
    np0 = np.arange(23)
    dataset0 = torch.utils.data.TensorDataset(torch.tensor(np0))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset0, num_replicas=3, rank=0, shuffle=True)
    train_loader = torch.utils.data.DataLoader(dataset0, batch_size=1, shuffle=False, sampler=sampler)
    print('WITHNOT call sampler.set_epoch()')
    for _ in range(4):
        print(np.concatenate([x[0].numpy() for x in train_loader]))
    print('WITH call sampler.set_epoch()')
    for ind_epoch in range(4):
        #it's necessary to use ind_epoch not some random number(otherwise, GPU will use same data)
        sampler.set_epoch(ind_epoch)
        print(np.concatenate([x[0].numpy() for x in train_loader]))


class MySimpleBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, batch_index):
        self.batch_index = batch_index
    def __len__(self):
        return len(self.batch_index)
    def set_epoch(self, epoch):
        pass
    def __iter__(self):
        yield from self.batch_index


class MyTrainBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, num_sample, batch_size, shuffle=True, random_seed=None, world_size=None, rank=None):
        if world_size is not None:
            assert (rank is not None) and rank<world_size
        else:
            world_size = 1
            rank = 0
        rand_generator = np.random.RandomState(random_seed)
        index_array = np.arange(num_sample)
        if num_sample%(batch_size*world_size)>0:
            tmp0 = batch_size*world_size - (num_sample%(batch_size*world_size))
            tmp1 = rand_generator.choice(index_array, size=(tmp0,), replace=False)
            index_array = np.concatenate([index_array, tmp1])
        index_array = index_array.reshape(world_size, -1, batch_size)
        self.num_sample = index_array.size
        self.num_batch = index_array.shape[1]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index_array = index_array
        self.rand_generator = rand_generator
        self.world_size = world_size
        self.rank = rank
    def __len__(self):
        return self.num_batch
    def set_epoch(self, epoch):
        pass
    def __iter__(self):
        if self.shuffle:
            tmp0 = self.index_array.reshape(-1)[self.rand_generator.permutation(self.num_sample)]
            tmp0 = tmp0.reshape(*self.index_array.shape)
        else:
            tmp0 = self.index_array
        yield from tmp0[self.rank].tolist()

def test_MyTrainBatchSampler():
    batch_size = 3
    world_size = 4
    num_batch = 5
    seed = 233
    kwargs = dict(num_sample=world_size*batch_size*num_batch, batch_size=batch_size, shuffle=True, random_seed=seed, world_size=world_size)

    z0 = [MyTrainBatchSampler(**kwargs, rank=x) for x in range(world_size)]
    for _ in range(3):
        assert {z for x in z0 for y in x for z in y}==set(range(world_size*batch_size*num_batch))

    z0 = MyTrainBatchSampler(**kwargs, rank=1)
    z1 = MyTrainBatchSampler(**kwargs, rank=1)
    for _ in range(3):
        assert tuple(y for x in z0 for y in x) == tuple(y for x in z1 for y in x)


def my_split_valdataset(index_list, batch_size, world_size=None, rank=None):
    hf_ceil = lambda x,y: (x-1)//y + 1
    if world_size is not None:
        assert world_size > 1
        assert rank is not None
        assert batch_size > 1 #could lead to len(batch)==0
        num_batch = hf_ceil(len(index_list), batch_size*world_size)
        batch_index = []
        for x in range(len(index_list)//(batch_size*world_size)):
            batch_index.append(index_list[((x*world_size+rank)*batch_size):((x*world_size+rank+1)*batch_size)])
        # separate last batch
        tmp0 = len(index_list)%(batch_size*world_size)
        if tmp0>0:
            tmp1 = tmp0//world_size + np.array([1]*(tmp0%world_size) + [0]*(world_size-(tmp0%world_size)))
            tmp2 = np.cumsum(np.concatenate([[0],tmp1]))
            tmp3 = index_list[-tmp0:]
            batch_index.append(tmp3[tmp2[rank]:tmp2[rank+1]])
        assert sum(len(x) for x in batch_index)==len(index_list[rank::world_size])
        # average last two batch two avoid len(batch)==0
        tmp0 = batch_index.pop(-1)
        tmp1 = batch_index.pop(-1)
        tmp0 = tmp0 + tmp1
        tmp1 = hf_ceil(len(tmp0), 2)
        batch_index.append(tmp0[:tmp1])
        batch_index.append(tmp0[tmp1:])
    else:
        num_batch = hf_ceil(len(index_list), batch_size)
        batch_index = [index_list[(x*batch_size):((x+1)*batch_size)] for x in range(num_batch)]
    assert len(batch_index)==num_batch
    assert all(len(x)>0 for x in batch_index)
    return batch_index
