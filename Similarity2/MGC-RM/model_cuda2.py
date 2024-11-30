import torch.nn
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
from scipy.special import comb
from torch_geometric.utils import k_hop_subgraph


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.W = nn.Linear(input_dim, hidden_dim, bias=False)
        # self.W = None
        self.q = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self,neighbor_embeddings):
        # self.W = nn.Linear(neighbor_embeddings.size(0), 3, bias=False)  # 动态输入层数
        attention_scores = self.q(F.relu(self.W(neighbor_embeddings)))
        attention_weights = F.softmax(attention_scores,dim=0)
        weighted_neighbor_embeddings = torch.sum(attention_weights * neighbor_embeddings, dim=0)
        return weighted_neighbor_embeddings

class NodeEmbeddingModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,args):
        super(NodeEmbeddingModule, self).__init__()
        torch.manual_seed(1234)
        self.args = args
        self.layers = nn.ModuleList([
            nn.Linear((2*input_dim)//(2**i), hidden_dim//(2**i)) for i in range(num_layers)
        ])
        self.attention_layers = nn.ModuleList([
            AttentionLayer(input_dim//(2**i), hidden_dim//(2**i)) for i in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim//(2**(num_layers-1)), output_dim)

    def forward(self, X_v, G):
        h_v = X_v
        """
        h_v[v] 1*2000 
        neighbor neighbors*2000
        attention_weights  neighbors*1
        h_N_v 邻居特征 1*2000
        [h_v, h_N_v] 1*4000
        h_v() 1*2000
        
        h_v[v] 1*2000 
        neighbor  neighbors*2000
        attention_weights  neighbors*1
        h_N_v 邻居特征 1*2000
        [h_v, h_N_v] 1*4000
        h_v() 1*1000
        neighbor  neighbors*1000
        attention_weights  neighbors*1
        h_N_v 邻居特征 1*1000
        [h_v, h_N_v] 1*2000
        h_v() 1*500
        """
        for l in range(len(self.layers)):   # 当前层
            h_N_v = []    #邻居节点的特征
            for v in range(len(X_v)):   # 遍历当前图中的每个节点  #15个
                # print('l', l)
                # print('v',v)
                neighbors = list(G.neighbors(v))   # 获取节点v的邻居
                # print('neighbors',neighbors)
                if not neighbors:  # 孤立节点
                    h_N_v.append(torch.zeros_like(h_v[v]))
                else:
                # 邻居节点特征
                    neighbor_embeddings = [h_v[i] for i in neighbors]
                # 连接邻居节点特征list
                    neighbor_embeddings_tensor = torch.stack(neighbor_embeddings,  dim=0)
                    # necighbor_embeddings_tensor = torch.tensor(np.array(neighbor_embeddings), dtype=torch.float32)
                    # print(neighbor_embeddings_tensor)
                    # print(h_v[v].unsqueeze(0))
                # 计算邻居节点特征attention
                    h_N_v_a = self.attention_layers[l](neighbor_embeddings_tensor)  # 每个节点的邻居嵌入hidden_dim//(2**i)
                    # print(h_N_v_a.size())
                    h_N_v.append(h_N_v_a)  # 添加每个节点的邻居嵌入
                    # print(len(h_N_v))
            # print('h_N_v', len(h_N_v))
            h_N_v = torch.stack(h_N_v,  dim=0) # list转torch 15个节点的邻居的嵌入
            # print('h_N_v',h_N_v.size())
            # print('h_v', h_v.size())
            # print('.', torch.cat([h_v, h_N_v], dim=1).size())
            # print(self.layers[l])
            h_v = F.relu(self.layers[l](torch.cat([h_v, h_N_v], dim=1)))
            # print(self.layers,self.attention_layers,self.output_layer)
        return self.output_layer(h_v[14])


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) #图的所有节点的特征，图的边索引
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.tanh(x)
        return x


class NodeEmbeddingModule2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, args):
        super(NodeEmbeddingModule2, self).__init__()
        torch.manual_seed(1234)
        self.args = args
        self.layers = nn.ModuleList([
            nn.Linear((2 * input_dim) // (2 ** i), hidden_dim // (2 ** i)) for i in range(num_layers)
        ])
        self.attention_layers = nn.ModuleList([
            GAT(input_dim // (2 ** i), input_dim // (2 ** i) ,input_dim // (2 ** i)) for i in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim // (2 ** (num_layers - 1)), output_dim)

    def forward(self, X_v, G):
        h_v = X_v
        """
        h_v[v] 1*2000 
        neighbor neighbors*2000
        attention_weights  neighbors*1
        h_N_v 邻居特征 1*2000
        [h_v, h_N_v] 1*4000
        h_v() 1*2000

        h_v[v] 1*2000 
        neighbor  neighbors*2000
        attention_weights  neighbors*1
        h_N_v 邻居特征 1*2000
        [h_v, h_N_v] 1*4000
        h_v() 1*1000
        neighbor  neighbors*1000
        attention_weights  neighbors*1
        h_N_v 邻居特征 1*1000
        [h_v, h_N_v] 1*2000
        h_v() 1*500
        """

        for l in range(len(self.layers)):  # 当前层
            h_N_v = []  # 邻居节点的特征
            neighbor_embeddings_tensor = h_v
            # # print(self.attention_layers)
            # mean_1 = torch.mean(neighbor_embeddings_tensor)  # 计算feature平均值
            # std_1 = torch.std(neighbor_embeddings_tensor)  # 计算feature标准差
            # neighbor_embeddings_tensor = (neighbor_embeddings_tensor - mean_1) / std_1  # 标准化特征 标准正态分布
            for v in range(len(X_v)):  # 遍历当前图中的每个节点  #15个
                neighbors = list(G.neighbors(v))
                edge_index = [(v, num) for num in neighbors]
                subgraph_edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                # print(subgraph_edge_index_tensor)
                if not neighbors:  # 孤立节点
                    h_N_v.append(torch.zeros_like(h_v[v]))
                else:
                    # 计算邻居节点特征
                    h_N_v_a = self.attention_layers[l](neighbor_embeddings_tensor,subgraph_edge_index_tensor)  # 每个节点的邻居嵌入hidden_dim//(2**i)
                    # print(h_N_v_a[v].size())  # 该节点对邻居节点的加权求和
                    h_N_v.append(h_N_v_a[v])  # 添加每个节点的邻居嵌入
                    # print(len(h_N_v))
            # print('h_N_v', len(h_N_v))
            h_N_v = torch.stack(h_N_v, dim=0)  # list转torch 15个节点的邻居的嵌入
            # print('h_N_v',h_N_v.size())
            # print('h_v', h_v.size())
            # print('.', torch.cat([h_v, h_N_v], dim=1).size())
            # print(self.layers[l])
            # print('[]',F.relu(self.layers[l](torch.cat([h_v[14].unsqueeze(0), h_N_v[14].unsqueeze(0)], dim=1))))
            h_v = F.relu(self.layers[l](torch.cat([h_v, h_N_v], dim=1)))
            # print('@',h_v[14])
            # print(h_v.size())
        return self.output_layer(h_v[14])




class RegressionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RegressionModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//5)
        self.fc3 = nn.Linear(hidden_dim//5, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()

class ILGRModel_test(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,args):
        super(ILGRModel_test, self).__init__()
        self.embedding_module = NodeEmbeddingModule2(input_dim, hidden_dim, output_dim, num_layers,args)
        self.regression_module = RegressionModule(output_dim, output_dim//2, 1)

    def forward(self, X_v, G):
        embeddings = self.embedding_module(X_v, G)
        scores = self.regression_module(embeddings)
        return scores

# 定义损失函数
def ranking_loss(scores1, true_ranks1):
    loss = 0
    # 归一化
    true_ranks = (true_ranks1 - torch.min(true_ranks1)) / (torch.max(true_ranks1) - torch.min(true_ranks1))
    scores = (scores1 - torch.min(scores1)) / (torch.max(scores1) - torch.min(scores1))
    for i in range(len(scores)-1):
        for j in range(i + 1, len(scores)-1):
            r_ij = true_ranks[i] - true_ranks[j]
            y_hat_ij = scores[i] - scores[j]
            f_r_ij = F.sigmoid(r_ij)
            loss += -f_r_ij * torch.log1p(F.sigmoid(y_hat_ij)-1) - (1 - f_r_ij) * torch.log1p(-F.sigmoid(y_hat_ij))

    return loss  /int(comb(len(scores), 2))


def softsort(x, tau=0.1):
    # 计算每对元素的差值
    pairwise_diff = x.unsqueeze(1) - x.unsqueeze(0)
    # 计算相似度矩阵
    similarity_matrix = -pairwise_diff.abs() / tau
    # 计算软排序矩阵
    soft_permutation_matrix = F.softmax(similarity_matrix, dim=1)
    # 计算软排序后的张量
    soft_sorted_x = soft_permutation_matrix @ x
    return soft_sorted_x



