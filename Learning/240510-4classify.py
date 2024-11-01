from torch.datasets import KarateClub  # dataset
from torch.utils import to_networkx
# matplotlib inline
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch.nn import Linear
from torch.nn import GCNConv
import time


def visualize_graph(G, color):
    plt.figure(figsize=(7, 7))
    plt.xticks([])  # Get or set the current tick locations and labels of the x-axis.
    plt.yticks([])  # Passing an empty list removes all x-ticks.
    nx.draw(G, pos=nx.spring_layout(G, seed=42), with_labels=True, node_color=color, cmap="Set2")
    ## nx.spring_layout ??
    #plt.show()


def visualize_embedding(h, color, epoch=None, loss=None):  ## 画点
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    #plt.show()


dataset = KarateClub()
print(f'Dataset: {dataset}:')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')  # 4 classify

data = dataset[0]  ## Get the first graph object. ???[0]
print(data)
"""
- edge_index：表示图的连接关系（start,end两个序列） torch.tensor([row, col]) 邻接矩阵
- x: 34*34 identity  原始图特征(初始化特征参数)，不断更新
- num_features：每个点的特征
- node labels：每个点的标签
- train_mask：有的节点木有标签（用来表示哪些节点要计算损失）???
"""
# print(f'features:\n\r {data.x}')
# print(f'train_mask:\n\r {data.train_mask}')  ##？？？？
edge_index = data.edge_index  # connect relation
print(edge_index.t())  # .t() Transpose the Tensor
# print(edge_index)
# print(f'num_features:\n\r {dataset.num_features}')


G = to_networkx(data, to_undirected=True)
visualize_graph(G, color=data.y)  ##??????


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)  ## How to set seed???
        # Set the CPU to generate a random number of seeds,
        # so that the results of the experiment can be reproduced next time.
        # The random number generated during model training is fixed,
        # so that the description effect can be approximated to the
        # greatest extent when the described model is reproduced.
        self.conv1 = GCNConv(dataset.num_features, 4)  # class GCNConv(in_channels: int, out_channels: int)
        # in_channels (int) – Size of each input sample
        self.conv2 = GCNConv(4, 4)  ## How to set channels number???
        self.conv3 = GCNConv(4, 2)  # out 2 to visual 2-D tensor
        self.classifier = Linear(2, dataset.num_classes)  # Applies a linear transformation to the incoming data
        ## 全连接，预测概率？

    def forward(self, x, edge_index):
        # print(x)
        h = self.conv1(x, edge_index)  ##?? def forward
        h = h.tanh()  ## ???
        # print(h)
        h = self.conv2(h, edge_index)  # Update the feature
        h = h.tanh()
        # print(h)
        h = self.conv3(h, edge_index)
        h = h.tanh()
        # print(h)

        # 分类层
        out = self.classifier(h)

        return out, h


model = GCN()
print(model)

## 随机初始化边关系？(未训练)
_, h = model(data.x, data.edge_index)
print(f'Embedding shape: {list(h.shape)}')
visualize_embedding(h, color=data.y)
plt.show()

criterion = torch.nn.CrossEntropyLoss() # Define loss criterion 交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.


def train(data):
    optimizer.zero_grad()  ## 梯度清零???
    out, h = model(data.x, data.edge_index)  # 是两维向量，主要是为了画图
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # semi-supervised  #只看有标签点的损失
    loss.backward()  ## 反向传播?
    optimizer.step()  ## 参数更新?
    return loss, h


for epoch in range(401):
    loss, h = train(data)
    if epoch % 10 == 0:
        visualize_embedding(h, color=data.y, epoch=epoch, loss=loss)
        time.sleep(0.3)

plt.show() ## ???
