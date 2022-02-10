import os
import torch
import pickle
import numpy as np

from torch.nn.parallel import DistributedDataParallel as DDP

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())

# TODO multi-node, local-rank, node_id


class MyModel00(torch.nn.Module):
    def __init__(self):
        super(MyModel00, self).__init__()
        self.fc0 = torch.nn.Linear(5, 13)
        self.fc1 = torch.nn.Linear(13, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = torch.nn.functional.relu(x)
        x = self.fc1(x)[:,0]
        return x


def demo_basic(rank, world_size):
    print(f'# demo_basic[rank={rank}]')
    torch.distributed.init_process_group(backend='nccl',
            init_method='tcp://127.0.0.1:23333', rank=rank, world_size=world_size)
    device = torch.device(f'cuda:{rank}')

    model = MyModel00().to(device)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(23, 5))
    labels = torch.randn(23).to(device)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    torch.distributed.destroy_process_group()


def _ddp_initialize_same_ddp(rank, world_size):
    print(f'## _ddp_initialize_same_ddp[rank={rank}]')
    torch.distributed.init_process_group(backend='nccl',
            init_method='tcp://127.0.0.1:23333', rank=rank, world_size=world_size)
    device = torch.device(f'cuda:{rank}')
    checkpoint0_path = hf_file(f'demo_checkpoint_model0_{rank}.pth')
    checkpoint_path = hf_file(f'demo_checkpoint_model_{rank}.pth')

    model = MyModel00().to(device) #different weight for different process
    torch.save(model.state_dict(), checkpoint0_path)
    ddp_model = DDP(model, device_ids=[rank]) #initialize the model at this step
    # ddp_model(torch.randn(23, 5)) #no need to explicitly pass one data to initialize the model
    torch.save(ddp_model.state_dict(), checkpoint_path)

def demo_ddp_initialize_same(world_size):
    print(f'# demo_ddp_initialize_same')
    torch.multiprocessing.spawn(_ddp_initialize_same_ddp, args=(world_size,), nprocs=world_size, join=True)
    torch_data = torch.randn(23, 5)

    ret0 = []
    for rank in range(world_size):
        with torch.no_grad():
            model = MyModel00()
            tmp0 = torch.load(hf_file(f'demo_checkpoint_model0_{rank}.pth'), map_location={f'cuda:{rank}':'cpu'})
            model.load_state_dict(tmp0)
            ret0.append(model(torch_data).numpy())
    ret0 = np.stack(ret0)
    assert np.abs(ret0 - ret0[0]).max() > 0.01 #they are different

    ret1 = []
    for rank in range(world_size):
        with torch.no_grad():
            model = MyModel00()
            tmp0 = torch.load(hf_file(f'demo_checkpoint_model_{rank}.pth'), map_location={f'cuda:{rank}':'cpu'})
            assert all(x.startswith('module.') for x in tmp0.keys())
            model.load_state_dict({x[7:]:y for x,y in tmp0.items()}) #remove 'module.'
            ret1.append(model(torch_data).numpy())
    ret1 = np.stack(ret1)
    assert np.abs(ret1 - ret0[0]).max() < 1e-5 #just broadcast the model[rank=0]
    assert np.abs(ret1 - ret1[0]).max() < 1e-5


def _ddp_sync_gradients_np_random_dataset(random_state=None):
    np_random_generator = np.random.RandomState(random_state)
    data = np_random_generator.randn(13, 5).astype(np.float32)
    label = np_random_generator.randn(13).astype(np.float32)
    return data, label

def _ddp_sync_gradients_ddp(rank, np_random_state_list):
    print(f'## _ddp_sync_gradients_ddp[rank={rank}]')
    world_size = len(np_random_state_list)
    torch.distributed.init_process_group(backend='nccl',
            init_method='tcp://127.0.0.1:23333', rank=rank, world_size=world_size)
    device = torch.device(f'cuda:{rank}')

    model = DDP(MyModel00().to(device), device_ids=[rank])
    tmp0 = torch.load(hf_file('demo_ddp_sync_gradients.pth'), map_location={'cpu': f'cuda:{rank}'})
    model.load_state_dict({('module.'+x):y for x,y in tmp0.items()})

    data,label = _ddp_sync_gradients_np_random_dataset(np_random_state_list[rank])
    data = torch.Tensor(data).to(device)
    label = torch.Tensor(label).to(device)

    model.zero_grad()
    predict = model(data)
    loss = torch.mean((predict-label)**2)
    loss.backward()
    # torch.distributed.barrier() #seems NO NEED to add barrier() to make sure gradients are synced
    with open(hf_file(f'_ddp_sync_gradients_ddp_{rank}.pkl'), 'wb') as fid:
        tmp0 = [x.grad.to('cpu').numpy().copy() for x in model.parameters()]
        pickle.dump(tmp0, fid)

    torch.distributed.destroy_process_group()

def demo_ddp_sync_gradients(world_size):
    print(f'# demo_ddp_sync_gradients')
    np_random_state_list = np.random.randint(1000, size=world_size)

    model = MyModel00()
    torch.save(model.state_dict(), hf_file('demo_ddp_sync_gradients.pth'))
    # model.load_state_dict(torch.load(hf_file('demo_ddp_sync_gradients.pth')))
    model.zero_grad()
    for random_state_i in np_random_state_list:
        data,label = _ddp_sync_gradients_np_random_dataset(random_state_i)
        data = torch.Tensor(data)
        predict = model(data)
        loss = torch.mean((predict-torch.Tensor(label))**2)
        loss.backward()
    grad_list0 = [x.grad.numpy().copy()/world_size for x in model.parameters()]

    torch.multiprocessing.spawn(_ddp_sync_gradients_ddp, args=(np_random_state_list,), nprocs=world_size, join=True)

    grad_list1 = []
    for rank in range(world_size):
        with open(hf_file(f'_ddp_sync_gradients_ddp_{rank}.pkl'), 'rb') as fid:
            grad_list1.append(pickle.load(fid))
    for grad_tensor_i in zip(*grad_list1):
        tmp0 = np.stack(grad_tensor_i)
        assert np.abs(tmp0-grad_tensor_i[0]).max() < 1e-4
    for x,y in zip(grad_list0, grad_list1[0]):
        assert np.abs(x-y).max() < 1e-4


class MyModel01(torch.nn.Module):
    def __init__(self):
        super(MyModel01, self).__init__()
        self.bn0 = torch.nn.BatchNorm1d(5, momentum=0.233, eps=0.00233)
        self.bn1 = torch.nn.BatchNorm1d(5, momentum=0.233, eps=0.00233)

    def forward(self, x):
        x = self.bn0(x)
        x = self.bn1(x).sum(dim=1)
        return x

def _ddp_bn_training_np_random_dataset(random_state=None):
    np_random_generator = np.random.RandomState(random_state)
    data = np_random_generator.randn(13, 5).astype(np.float32)
    label = np_random_generator.randn(13).astype(np.float32)
    return data, label

def _ddp_bn_training_ddp(rank, np_random_state_list):
    print(f'## _ddp_bn_training_ddp[rank={rank}]')
    world_size = len(np_random_state_list)
    torch.distributed.init_process_group(backend='nccl',
            init_method='tcp://127.0.0.1:23333', rank=rank, world_size=world_size)
    device = torch.device(f'cuda:{rank}')

    model = DDP(MyModel01().to(device), device_ids=[rank])
    tmp0 = torch.load(hf_file('demo_ddp_bn_training.pth'), map_location={'cpu': f'cuda:{rank}'})
    model.load_state_dict({('module.'+x):y for x,y in tmp0.items()})

    np_random_state0 = np_random_state_list[0] if rank==0 else np.random.randint(10000)
    np_random_state1 = np_random_state_list[rank]

    data,_ = _ddp_bn_training_np_random_dataset(np_random_state0)
    model(torch.Tensor(data).to(device))
    data,_ = _ddp_bn_training_np_random_dataset(np_random_state1)
    model(torch.Tensor(data).to(device))
    # torch.distributed.barrier() #maybe need to add barrier() to make sure gradients are synced
    torch.save(model.state_dict(), hf_file(f'demo_ddp_bn_training_{rank}.pth'))

    torch.distributed.destroy_process_group()

def demo_ddp_bn_training(world_size):
    print('# demo_ddp_bn_training')
    np_random_state_list = np.random.randint(1000, size=world_size)

    model = MyModel01()
    torch.save(model.state_dict(), hf_file('demo_ddp_bn_training.pth'))
    bn_state0 = []
    for rank in range(world_size):
        model.load_state_dict(torch.load(hf_file('demo_ddp_bn_training.pth')))
        data,_ = _ddp_bn_training_np_random_dataset(np_random_state_list[0])
        model(torch.Tensor(data))
        data,_ = _ddp_bn_training_np_random_dataset(np_random_state_list[rank])
        model(torch.Tensor(data))
        bn_state0.append([y.numpy().copy() for x in (model.bn0,model.bn1) for y in (x.running_mean,x.running_var)])

    torch.multiprocessing.spawn(_ddp_bn_training_ddp, args=(np_random_state_list,), nprocs=world_size, join=True)

    bn_state1 = []
    for rank in range(world_size):
        tmp0 = torch.load(hf_file(f'demo_ddp_bn_training_{rank}.pth'), map_location={f'cuda:{rank}': 'cpu'})
        bn_state1.append([
            tmp0['module.bn0.running_mean'].numpy().copy(),
            tmp0['module.bn0.running_var'].numpy().copy(),
            tmp0['module.bn1.running_mean'].numpy().copy(),
            tmp0['module.bn1.running_var'].numpy().copy(),
        ])
    tmp0 = ((y0,y1) for x0,x1 in zip(bn_state0) for y0,y1 in zip(x0,x1))
    for x0,x1 in zip(bn_state0,bn_state1):
        for y0,y1 in zip(x0,x1):
            assert np.abs(y0-y1).max() < 1e-5


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()

    # see https://pytorch.org/docs/stable/nn.html#distributeddataparallel
    torch.multiprocessing.set_start_method('spawn')

    if n_gpus>1:
        world_size = n_gpus
        torch.multiprocessing.spawn(demo_basic, args=(world_size,), nprocs=world_size, join=True)

    if n_gpus>1:
        demo_ddp_initialize_same(n_gpus)

    if n_gpus>1:
        demo_ddp_sync_gradients(n_gpus)

    if n_gpus>1:
        demo_ddp_bn_training(n_gpus)
