import torch
import torch.nn as nn
from model import GAT
import torch.nn.functional as F
from param_parser import parameter_parser
from torchinfo import summary
from torch_geometric.nn import global_mean_pool, global_add_pool


class MFC_RMF(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, args):
        super(MFC_RMF, self).__init__()
        self.args = args
        # get model
        self.gat = GAT(input_size, hidden_size, output_size)
        self.readout_rep = readout(output_size)
        self.readout_match = readout(args.match_size)
        self.match = matching(output_size, args.match_size)
        self.mlp = MLP(args.match_size)


    # 自表示操作：输出自表示特征
    def self_representation(self, x, edge_index):
        h_self = self.gat(x, edge_index)
        return h_self

    # 互表示操作：输出互表示特征
    def mutual_representation(self, h_1, h_2):
        cos_sim = torch.cosine_similarity(h_1.unsqueeze(1), h_2.unsqueeze(0), dim=-1)
        cos_ceo = cos_sim / torch.sum(cos_sim, dim=0)
        h_mutual = torch.mm(cos_ceo, h_2)
        return h_mutual

    def forward(self, graph_pair):
        # print(graph_pair)
        # graph_pair = torch.load(graph_pair)
        # 图对信息
        edge_index_1 = graph_pair.edge_index_s
        edge_index_2 = graph_pair.edge_index_t


        feature_1 = torch.tensor(graph_pair.x_s, dtype=torch.float32)
        mean_1 = torch.mean(feature_1)
        std_1 = torch.std(feature_1)
        feature_1 = (feature_1 - mean_1) / std_1

        feature_2 = torch.tensor(graph_pair.x_t, dtype=torch.float32)
        mean_2 = torch.mean(feature_2)
        std_2 = torch.std(feature_2)
        feature_2 = (feature_2 - mean_2) / std_2


        label = graph_pair.label
        label_exp = torch.exp(-1 * label)

        # 多粒度交叉表示
        # 局部自表示
        h_1_self = self.self_representation(feature_1, edge_index_1)
        h_2_self = self.self_representation(feature_2, edge_index_2)
        # 全局自表示
        h_1_self_global = self.readout_rep(h_1_self)
        # print('h_1_self_global.shape:', h_1_self_global.shape)
        h_2_self_global = self.readout_rep(h_2_self)

        # 局部互表示
        h_1_mutual = self.mutual_representation(h_1_self, h_2_self)
        h_2_mutual = self.mutual_representation(h_2_self, h_1_self)
        # 全局互表示
        h_1_mutual_global = self.readout_rep(h_1_mutual)
        h_2_mutual_global = self.readout_rep(h_2_mutual)


        # 多粒度交叉匹配
        # 局部-局部匹配
        miu_1 = self.match(h_1_self, h_1_mutual)
        miu_2 = self.match(h_2_self, h_2_mutual)
        # 读出全局匹配向量
        miu_1_global = self.readout_rep(miu_1)
        miu_2_global = self.readout_rep(miu_2)


        # 局部-全局匹配
        # print('---------------------------')
        # print(h_1_mutual_global)
        # print(h_1_mutual_global.shape)
        # print('---------------------------')
        phi_1 = self.match(h_1_self, h_1_mutual_global)
        phi_2 = self.match(h_2_self, h_2_mutual_global)
        # 读出全局匹配向量
        phi_1_global = self.readout_rep(phi_1)
        phi_2_global = self.readout_rep(phi_2)


        # 全局-局部匹配
        psi_1 = self.match(h_1_self_global, h_1_mutual)
        psi_2 = self.match(h_2_self_global, h_2_mutual)
        # 读出全局匹配向量
        psi_1_global = self.readout_rep(psi_1)
        psi_2_global = self.readout_rep(psi_2)


        # 全局-全局匹配
        omega_1 = self.match(h_1_self_global, h_1_mutual_global)
        omega_2 = self.match(h_2_self_global, h_2_mutual_global)
        # # 读出全局匹配向量
        # omega_1_global = self.readout_rep(omega_1)
        # omega_2_global = self.readout_rep(omega_2)


        # 匹配特征融合
        z_1 = torch.cat([miu_1_global, phi_1_global, psi_1_global, omega_1], dim=1).t()
        # z_1 = self.mlp(z_1)
        # print(z_1.shape)
        z_2 = torch.cat([miu_2_global, phi_2_global, psi_2_global, omega_2], dim=1).t()
        z = torch.cat([z_1, z_2], dim=0)
        # print(z.shape)
        z = self.mlp(z.t())

        return z, label, label_exp








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


# 匹配操作：输出匹配特征
class matching(nn.Module):
    def __init__(self, output_size, match_size):
        super(matching, self).__init__()
        self.output_size = output_size
        self.match_size = match_size
        self.weight = torch.nn.Parameter(torch.Tensor(self.output_size, self.match_size))
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, h_1, h_2):
        # print(h_1.shape)
        # print(h_2.shape)
        h_1 = h_1.unsqueeze(1)
        h_2 = h_2.unsqueeze(1)
        weight = self.weight.unsqueeze(0)
        h_1_w = torch.mul(h_1, weight)
        h_2_w = torch.mul(h_2, weight)
        match = torch.cosine_similarity(h_1_w, h_2_w)
        return match



# 融合操作：输出融合特征
class MLP(nn.Module):
    def __init__(self, match_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(match_size * 8, match_size * 2)
        nn.init.xavier_uniform_(self.linear1.weight.data)
        nn.init.zeros_(self.linear1.bias.data)

        self.linear2 = nn.Linear(match_size * 2, match_size)
        nn.init.xavier_uniform_(self.linear2.weight.data)
        nn.init.zeros_(self.linear2.bias.data)

        self.linear3 = nn.Linear(match_size, match_size // 2)
        nn.init.xavier_uniform_(self.linear3.weight.data)
        nn.init.zeros_(self.linear3.bias.data)

        self.linear4 = nn.Linear(match_size // 2, 1)
        nn.init.xavier_uniform_(self.linear4.weight.data)
        nn.init.zeros_(self.linear4.bias.data)

        self.relu = nn.ReLU()

    def forward(self, z):
        z = self.linear1(z)
        z = self.relu(z)
        z = self.linear2(z)
        z = self.relu(z)
        z = self.linear3(z)
        z = self.relu(z)
        z = self.linear4(z)
        z = F.sigmoid(z)
        return z


args = parameter_parser()

model = MFC_RMF(args.input_dim, args.hidden_dim, args.output_dim, args)
print(summary(model))

