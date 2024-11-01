import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from IPython.display import Javascript


dataset = TUDataset(root='data/TUDataset', name='MUTAG')  # 加载数据集

print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # 得到数据中的第一个图

print()
print(data)
print('=============================================================')

# 获得图的一些统计特征
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        # 使用 GCNConv
        # 为了让模型更稳定我们也可以使用带有跳跃链接的 GraphConv
        # from torch_geometric.nn import GraphConv
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. 获得节点的嵌入
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. 读出层
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. 应用最后的分类器
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


model = GCN(hidden_channels=64)
print(model)

display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

model = GCN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()

    for data in train_loader:  # 迭代获得各个批数据
         out = model(data.x, data.edge_index, data.batch)  # 前向传播
         loss = criterion(out, data.y)  # 计算损失
         loss.backward()  # 反向传播
         optimizer.step()  # 参数更新
         optimizer.zero_grad()  # 梯度清零


def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # 迭代获得各个批数据
         out = model(data.x, data.edge_index, data.batch)
         pred = out.argmax(dim=1)  # 取最大概率的类作为预测
         correct += int((pred == data.y).sum())  # 与真实标签做比较
     return correct / len(loader.dataset)  # 计算准确率


for epoch in range(1, 171):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
