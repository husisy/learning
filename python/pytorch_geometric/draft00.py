import os
import shutil
import torch
import torch_geometric
import torch_scatter

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
edge_index = torch.tensor([[0,1],[1,0],[1,2],[2,1]], dtype=torch.int64).T.contiguous()
data = torch_geometric.data.Data(x=x, edge_index=edge_index)
data.x
data.edge_index
data.num_nodes
data.num_edges
data.num_node_features
data.contains_isolated_nodes()
data.contains_self_loops()
data.is_directed()
# data = data.to(torch.device('cuda'))


dataset = torch_geometric.datasets.TUDataset('/opt/pytorch_data', name='ENZYMES')
len(dataset) #600
dataset.num_classes #6
dataset.num_node_features #3
data.is_undirected() #True
dataset[:540]
set([x.y.item() for x in dataset])
data = dataset[0] #Data(edge_index=[2, 168], x=[37, 3], y=[1])
data.y
# dataset.shuffle()
# dataset = dataset[torch.randperm(len(dataset))]

dataset = torch_geometric.datasets.Planetoid(root='/opt/pytorch_data', name='Cora')
len(dataset) #1, only 1 data
dataset.num_classes #7
dataset.num_node_features #1433
data = dataset[0] #Data(edge_index=[2,10556], test_mask=[2708], train_mask=[2708], val_mask=[2708], x=[2708,1433], y=[2708])
data.is_undirected() #True
data.train_mask #(torch,bool,(2708,))
data.train_mask.sum().item() #140
data.val_mask.sum().item() #500
data.test_mask.sum().item() #1000

# see https://stackoverflow.com/a/28052583
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
shapenet_ROOT = '/opt/pytorch_data/ShapeNet'
shapenet_processed_dir = os.path.join(shapenet_ROOT,'processed')
dataset = torch_geometric.datasets.ShapeNet(root=shapenet_ROOT, categories=['Airplane'])
len(dataset) #2346
dataset.num_classes #50
dataset.num_node_features #3
data = dataset[0] #Data(category=[1], pos=[2518, 3], x=[2518, 3], y=[2518])

transform = torch_geometric.transforms.KNNGraph(k=6)
# torch_geometric.transforms.RandomTranslate(0.01)
dataset = torch_geometric.datasets.ShapeNet(root=shapenet_ROOT, categories=['Airplane'], transform=transform)
if os.path.exists(shapenet_processed_dir):
    shutil.rmtree(shapenet_processed_dir)
dataset = torch_geometric.datasets.ShapeNet(root=shapenet_ROOT, categories=['Airplane'], pre_transform=transform)

def np_batch_to_index(np0, num_batch):
    assert np.all(0<=np0) and np.all(np0<num_batch)
    assert np0.ndim==1
    ret = []
    for x in range(num_batch):
        ind0 = np.nonzero(np0==x)[0]
        if len(ind0):
            assert len(ind0)==(ind0[-1]+1-ind0[0])
            ret.append((ind0[0],ind0[-1]+1))
        else:
            ret.append((0,0))
    return ret

dataset = torch_geometric.datasets.TUDataset('/opt/pytorch_data', name='ENZYMES', use_node_attr=True)
dataset.num_node_features #21
dataloader = torch_geometric.data.DataLoader(dataset, batch_size=32, shuffle=True)
data = next(iter(dataloader)) #Batch(batch=[1004], edge_index=[2, 3846], x=[1004, 21], y=[32])
data.num_graphs #32
data.batch #(torch,int64,(1004,))
ret_ = torch_scatter.scatter_mean(data.x, data.batch, dim=0) #(torch,float32,(32,21))
ind0 = np_batch_to_index(data.batch.numpy(), data.num_graphs)
ret0 = torch.stack([data.x[x:y].mean(dim=0) for x,y in ind0])
assert hfe(ret_.numpy(), ret0.numpy()) < 1e-5
