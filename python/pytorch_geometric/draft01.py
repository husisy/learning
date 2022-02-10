import torch
import torch.nn.functional as F
import torch_geometric
import torch_scatter
from tqdm import tqdm

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


class Net(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(Net, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(num_node_features, 16)
        self.conv2 = torch_geometric.nn.GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


device = torch.device('cuda')
dataset = torch_geometric.datasets.Planetoid(root='/opt/pytorch_data', name='Cora')

model = Net(dataset.num_node_features, dataset.num_classes).to(device)
data = dataset[0].to(device) #only one data
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in tqdm(range(200)):
    optimizer.zero_grad()
    predict = model(data)
    loss = F.cross_entropy(predict[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
accuracy = (pred[data.test_mask] == data.y[data.test_mask]).cpu().numpy().mean()
print('accuracy:', accuracy) #around 0.8
