import torch.nn
from torch import nn
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



class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.encoder = nn.Sequential(
            nn.Linear(6, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 2),
            nn.LeakyReLU(),
            nn.Linear(2, 1),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 2),
            nn.LeakyReLU(),
            nn.Linear(2, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 6),
            nn.Tanh()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded



# 全局读出操作：输出全局特征
class readout(nn.Module):
    def __init__(self, output_size):
        super(readout, self).__init__()
        self.output_size = output_size
        self.weight = torch.nn.Parameter(torch.Tensor(self.output_size, self.output_size))
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, h):
        global_h = torch.matmul(torch.mean(h, dim=0), self.weight)
        transformed_global = torch.tanh(global_h)
        sigmoid_scores = torch.sigmoid(torch.mm(h, transformed_global.view(-1, 1)).sum(dim=1))
        h_global = sigmoid_scores.unsqueeze(-1) * h
        return h_global.sum(dim=0).unsqueeze(0)





class AdditiveAttentionReadout(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AdditiveAttentionReadout, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # 定义用于计算注意力得分的前馈神经网络
        self.attn = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        # 用于转换全局特征的权重
        self.fc = nn.Linear(input_size, output_size)
        nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, h):
        """
        :param h: 节点特征，形状为 [num_nodes, input_size]
        :return: 图级别的特征，形状为 [1, output_size]
        """
        num_nodes = h.size(0)
        # 初始化一个隐藏状态作为查询向量
        query = torch.zeros(1, self.hidden_size).to(h.device)
        # 扩展节点特征和查询向量以匹配维度
        h_expanded = h.unsqueeze(1).expand(num_nodes, num_nodes, self.input_size)
        query_expanded = query.expand(num_nodes, self.hidden_size).unsqueeze(0)
        # 计算注意力得分
        attn_scores = self.attn(torch.cat([h_expanded, query_expanded], dim=-1)).squeeze(-1)
        # 使用softmax将得分转换为注意力权重
        attn_weights = F.softmax(attn_scores, dim=1)
        # 计算加权的节点特征
        weighted_h = torch.bmm(attn_weights.unsqueeze(1), h.unsqueeze(0)).squeeze(1)
        # 转换加权后的节点特征以获得最终的图级别特征
        global_feature = self.fc(weighted_h)
        return global_feature


class GraphEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GraphEncoder, self).__init__()
        self.gcn = nn.Linear(input_size, hidden_size)
        self.readout = readout(hidden_size, hidden_size, output_size)

    def forward(self, h, adj):
        h = F.relu(self.gcn(h))
        graph_feature = self.readout(h)
        return graph_feature


class GraphDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GraphDecoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, graph_feature):
        h = F.relu(self.fc1(graph_feature))
        reconstructed_h = self.fc2(h)
        return reconstructed_h


class GraphAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size,output_size):
        super(GraphAutoencoder, self).__init__()
        self.encoder = GraphEncoder(input_size, hidden_size, output_size)
        self.decoder = GraphDecoder(output_size, hidden_size, input_size)

    def forward(self, h, adj):
        graph_feature = self.encoder(h, adj)
        reconstructed_h = self.decoder(graph_feature)
        return reconstructed_h


class ProjectionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x