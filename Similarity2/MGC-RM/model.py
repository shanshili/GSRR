import torch.nn
import torch_geometric as pyg
import torch_geometric.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) #edge_index表示邻接矩阵？
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.tanh(x)
        return x


