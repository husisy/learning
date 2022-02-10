'''demo single node, multiple GPU training'''
import os
import sys #redirect stdout
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torch.distributed as dist
from tqdm import tqdm
from collections import defaultdict


torch.backends.cudnn.benchmark = True
# TODO maybe DistributedSampler is necessary, shall we ignore that in validation phase
# TODO PIL data corrupt

# python main.py --arch resnet50 --dist-url 'tcp://127.0.0.1:23333'
#        --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ~/pytorch_data/ILSVRC2012


def adjust_learning_rate(optimizer, ind_epoch, lr0, lr1, warmstart_epoch):
    if ind_epoch < warmstart_epoch:
        lr = lr0 * (lr1/lr0)**(ind_epoch/warmstart_epoch)
    else:
        lr = lr1 * 0.1**((ind_epoch-warmstart_epoch)//30)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def topk_accuracy(predict, label, topk=(1,)):
    with torch.no_grad():
        tmp0 = predict.topk(max(topk), dim=1)[1]
        correct = (tmp0==label.view(-1,1))
        ret = [correct[:,:x].sum().item() for x in topk]
    return ret


def main_worker(gpu_i, args):
    args['gpu_i'] = gpu_i
    stdout_fid = open('stdout-rank{}.log'.format(args['gpu_i']), 'w')
    sys.stdout = stdout_fid
    assert args['num_node']==1 and args['node_rank']==0
    args['rank'] = args['gpu_i'] + args['node_rank'] * args['num_gpu_per_node']
    args['world_size'] = args['num_node'] * args['num_gpu_per_node']
    dist.init_process_group(backend='nccl', init_method=args['dist_url'], world_size=args['world_size'], rank=args['rank'])
    model = torchvision.models.resnet50()
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    torch.cuda.set_device(args['gpu_i'])
    device = torch.device('cuda:{}'.format(args['gpu_i']))
    model.to(device=device) #in-place operation, also return value
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args['gpu_i']])
    optimizer = torch.optim.SGD(model.parameters(), args['lr0'], momentum=args['momentum'], weight_decay=args['weight_decay'])

    normalize = torchvision.transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    tmp0 = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize,
    ])
    trainset = torchvision.datasets.ImageFolder(os.path.join(args['datadir'], 'train'), transform=tmp0)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size_per_gpu'],
            shuffle=False, num_workers=args['num_worker_per_gpu'], pin_memory=True, sampler=train_sampler)

    tmp0 = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize,
    ])
    validationset = torchvision.datasets.ImageFolder(os.path.join(args['datadir'], 'val'), transform=tmp0)
    # TODO mpi all reduce
    # TODO torch.utils.data.distributed.DistributedSampler add tag indicate non-duplicated sample
    # TODO pytorch rpc framework
    validationloader = torch.utils.data.DataLoader(validationset, batch_size=args['batch_size_per_gpu'],
            shuffle=False, num_workers=args['num_worker_per_gpu'], pin_memory=True)

    metric_history = defaultdict(list)
    for ind_epoch in range(args['num_epoch']):
        train_sampler.set_epoch(ind_epoch)
        adjust_learning_rate(optimizer, ind_epoch, args['lr0'], args['lr1'], args['warmstart_epoch'])
        with tqdm(total=len(trainloader), file=sys.stdout, dynamic_ncols=True,
                    desc='rank-{}/epoch-{}'.format(args['rank'], ind_epoch)) as pbar:
            model.train()
            train_top1_correct = 0
            train_top5_correct = 0
            train_total = 0
            for ind_batch, (data_i, label_i) in enumerate(trainloader):
                data_i, label_i = data_i.to(device, non_blocking=True), label_i.to(device, non_blocking=True)
                predict = model(data_i)
                tmp0 = topk_accuracy(predict, label_i, topk=(1,5))
                train_top1_correct += tmp0[0]
                train_top5_correct += tmp0[1]
                train_total += label_i.size()[0]
                loss = F.cross_entropy(predict, label_i)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                metric_history['train-loss'].append(loss.item())
                if ind_batch+1 < len(trainloader):
                    pbar.set_postfix({
                        'loss':'{:5.3}'.format(loss.item()),
                        'top1-acc':'{:4.3}%'.format(100*train_top1_correct/train_total),
                        'top5-acc':'{:4.3}%'.format(100*train_top5_correct/train_total),
                    })
                    pbar.update() #move the last update to val
            metric_history['train-top1-acc'].append(train_top1_correct / train_total)
            metric_history['train-top5-acc'].append(train_top5_correct / train_total)
            # lr_scheduler.step()

            model.eval()
            validation_top1_correct = 0
            validation_top5_correct = 0
            validation_total = 0
            with torch.no_grad():
                # TODO remove duplicated sample in last batch
                for data_i,label_i in validationloader:
                    data_i, label_i = data_i.to(device, non_blocking=True), label_i.to(device, non_blocking=True)
                    predict = model(data_i)
                    tmp0 = topk_accuracy(predict, label_i, topk=(1,5))
                    validation_top1_correct += tmp0[0]
                    validation_top5_correct += tmp0[1]
                    validation_total += label_i.size()[0]
            metric_history['validation-top1-acc'].append(validation_top1_correct / validation_total)
            metric_history['validation-top5-acc'].append(validation_top5_correct / validation_total)
            pbar.set_postfix({
                'top1-acc':'{:4.3}%'.format(100*train_top1_correct/train_total),
                'top5-acc':'{:4.3}%'.format(100*train_top5_correct/train_total),
                'val-top1-acc':'{:4.3}%'.format(100*validation_top1_correct/validation_total),
                'val-top5-acc':'{:4.3}%'.format(100*validation_top5_correct/validation_total),
            })
            pbar.update()
    with open('metric_history_rank{}.pkl'.format(args['gpu_i']), 'wb') as fid:
        pickle.dump(metric_history, fid)
    stdout_fid.close()


if __name__=='__main__':
    # TODO batch-size
    # TODO learning-rate
    args = {
        'dist_url': 'tcp://127.0.0.1:23333',
        'num_node': 1,
        'node_rank': 0,
        'num_gpu_per_node': 4,

        'datadir': '~/pytorch_data/ILSVRC2012',
        'num_worker_per_gpu': 2, #data loading workers per GPU
        'num_epoch': 90,
        'batch_size_per_gpu': 208,
        'lr0': 0.3,
        'lr1': 0.3,
        'warmstart_epoch': 5,
        'momentum': 0.9,
        'weight_decay': 1e-4,
    }
    torch.multiprocessing.spawn(main_worker, nprocs=args['num_gpu_per_node'], args=(args,))
