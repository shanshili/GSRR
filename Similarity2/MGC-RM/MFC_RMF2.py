import numpy as np
import torch
import torch.nn as nn
from numpy.random import randn

from model import GAT
# from param_parser import parameter_parser
from torchinfo import summary
from torch_geometric.nn import global_mean_pool, global_add_pool
import argparse
import networkx as nx
from torch.optim import Adam
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time


class MFC_RMF(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, args):
        super(MFC_RMF, self).__init__()
        torch.manual_seed(1234)
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

        # 标准化
        # feature_1 = torch.tensor(graph_pair.x_s, dtype=torch.float32)
        feature_1 = graph_pair.x_s.clone().detach()
        mean_1 = torch.mean(feature_1) # 计算feature平均值
        std_1 = torch.std(feature_1)  # 计算feature标准差
        feature_1 = (feature_1 - mean_1) / std_1  # 标准化特征 标准正态分布

        # feature_2 = torch.tensor(graph_pair.x_t, dtype=torch.float32)
        feature_2 = graph_pair.x_t.clone().detach()
        mean_2 = torch.mean(feature_2)
        std_2 = torch.std(feature_2)
        feature_2 = (feature_2 - mean_2) / std_2


        label = graph_pair.label
        label_exp = torch.exp(-1 * label) # 无用

        # 多粒度交叉表示
        # 局部自表示
        h_1_self = self.self_representation(feature_1, edge_index_1)
        h_2_self = self.self_representation(feature_2, edge_index_2)
        #print('self_re finish \n')
        # 全局自表示
        h_1_self_global = self.readout_rep(h_1_self)
        # print('h_1_self_global.shape:', h_1_self_global.shape)
        h_2_self_global = self.readout_rep(h_2_self)
        #print('self_global_re finish')

        # 局部互表示
        h_1_mutual = self.mutual_representation(h_1_self, h_2_self)
        h_2_mutual = self.mutual_representation(h_2_self, h_1_self)
        #print('mutual_re finish \n')
        # 全局互表示
        h_1_mutual_global = self.readout_rep(h_1_mutual)
        h_2_mutual_global = self.readout_rep(h_2_mutual)
        #print('mutual_global_re finish \n')


        # 多粒度交叉匹配
        # 局部-局部匹配
        miu_1 = self.match(h_1_self, h_1_mutual)
        miu_2 = self.match(h_2_self, h_2_mutual)
        #print('miu finish \n')
        # 读出全局匹配向量
        miu_1_global = self.readout_rep(miu_1)
        miu_2_global = self.readout_rep(miu_2)
        #print('miu_global finish')


        # 局部-全局匹配
        # print('---------------------------')
        # print(h_1_mutual_global)
        # print(h_1_mutual_global.shape)
        # print('---------------------------')
        phi_1 = self.match(h_1_self, h_1_mutual_global)
        phi_2 = self.match(h_2_self, h_2_mutual_global)
        #print('phi finish \n')
        # 读出全局匹配向量
        phi_1_global = self.readout_rep(phi_1)
        phi_2_global = self.readout_rep(phi_2)
        #print('phi_global finish \n')


        # 全局-局部匹配
        psi_1 = self.match(h_1_self_global, h_1_mutual)
        psi_2 = self.match(h_2_self_global, h_2_mutual)
        #print('psi finish \n')
        # 读出全局匹配向量
        psi_1_global = self.readout_rep(psi_1)
        psi_2_global = self.readout_rep(psi_2)
        #print('psi_global finish \n')

        # 全局-全局匹配
        omega_1 = self.match(h_1_self_global, h_1_mutual_global)
        omega_2 = self.match(h_2_self_global, h_2_mutual_global)
        # # 读出全局匹配向量
        # omega_1_global = self.readout_rep(omega_1)
        # omega_2_global = self.readout_rep(omega_2)
        #print('omega finish \n')

        # 匹配特征融合
        z_1 = torch.cat([miu_1_global, phi_1_global, psi_1_global, omega_1], dim=1).t()
        # z_1 = self.mlp(z_1)
        # print(z_1.shape)
        z_2 = torch.cat([miu_2_global, phi_2_global, psi_2_global, omega_2], dim=1).t()
        z = torch.cat([z_1, z_2], dim=0)
        # print(z.shape)
        z = self.mlp(z.t()) # 相似度分数？？

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

class GraphPair:
    def __init__(self, edge_index_s, edge_index_t, x_s, x_t, label):  # edge_index_s, edge_index_t,
        # self.adj = adj
        # self.adj_p = adj_p
        self.edge_index_s = edge_index_s
        self.edge_index_t = edge_index_t
        self.x_s = x_s
        self.x_t = x_t
        self.label = label

    def save(self, GraphPair_file):
        """
        Save the graph pair to a file using PyTorch's serialization.
        """
        torch.save({
            # 'adj': self.adj,
            # 'adj_p': self.adj_p,
            'edge_index_s': self.edge_index_s,
            'edge_index_t': self.edge_index_t,
            'x_s': self.x_s,
            'x_t': self.x_t,
            'label': self.label,
        }, GraphPair_file)

    @staticmethod
    def load(GraphPair_file):
        """
        Load a graph pair from a file.
        """
        loaded_data = torch.load(GraphPair_file)
        return GraphPair(
            # adj=loaded_data['adj'],
            # adj_p=loaded_data['adj_p'],
            edge_index_s=loaded_data['edge_index_s'],
            edge_index_t=loaded_data['edge_index_t'],
            x_s = loaded_data['x_s'],
            x_t = loaded_data['x_t'],
            label = loaded_data['label']
        )




parser = argparse.ArgumentParser(
    description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
# parser.add_argument("--name", type=str, default="Citeseer")
parser.add_argument("--max_epoch", type=int, default=21)
parser.add_argument("--lr", type=float, default=0.0001)
#parser.add_argument("--n_clusters", default=6, type=int)
parser.add_argument("--hidden_dim", default=1000, type=int)
parser.add_argument("--output_dim", default=250, type=int)
parser.add_argument("--match_size", default=250, type=int)  #######？
parser.add_argument("--weight_decay", type=int, default=1e-4)
#parser.add_argument("--alpha", type=float, default=0.2, help="Alpha for the leaky_relu.")
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print("use cuda: {}".format(args.cuda))
device = torch.device("cuda" if args.cuda else "cpu")


'''
setting processing data size
data_num: number of feature
'''
node = 450
data_num = 2000
adj_ori = np.load('./origin_data/adj_ori'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', allow_pickle=True)
fea_ori = np.load('./origin_data/fea_ori'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', allow_pickle=True)
perturbed_graph = np.load('./perturbation_data/perturbed_graph'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz')
perturbed_label = np.load('./perturbation_data/perturbed_label'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz')
perturbed_fea = np.load('./perturbation_data/perturbed_fea'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz')
print('perturbed_graph',perturbed_graph.files,perturbed_graph['arr_0'].shape)
# print('perturbed_label',perturbed_label.files,perturbed_label['arr_0'].shape)
# print('perturbed_fea',perturbed_fea.files,perturbed_fea['arr_0'].shape)
perturbed_a = perturbed_graph['arr_0']
perturbed_l = perturbed_label['arr_0']
perturbed_fe = perturbed_fea['arr_0']
# print('adj_ori:\n',adj_ori.files,'\n',adj_ori['arr_0'])
# print(type(adj_ori['arr_0']),adj_ori['arr_0'].shape,adj_ori['arr_0'])

G_o = nx.Graph(adj_ori['arr_0'])
edge_index_s = np.array(list(G_o.edges)).T
# print('edge_index_s:\n',edge_index_s)
# print('edge_index_s_type:\n',type(edge_index_s))
# print('edge_index_s_type:\n',edge_index_s.T.shape)


'''
setting processing graph pair number
i range in 0-(node-1)
'''
j = 310  # 图对编号
# for i in range(perturbed_a.shape[0]):
print('======= perturbed：',j,'========================')
perturbed_graph_a = perturbed_a[j]
# perturbed_graph_l = perturbed_l[i]
perturbed_f = perturbed_fe[j]
perturbed_graph = nx.Graph(perturbed_graph_a)
edge_index_t = np.array(list(perturbed_graph.edges)).T

# 转换成 PyTorch 张量
# adj_tensor = torch.tensor(adj_ori['arr_0'], dtype=torch.int64)
# adj_p_tensor = torch.tensor(perturbed_graph_a, dtype=torch.int64)
x_s_tensor = torch.tensor(fea_ori['arr_0'], dtype=torch.float32)
x_t_tensor = torch.tensor(perturbed_f, dtype=torch.float32)
#print(fea_ori['arr_0'].shape)
#print(perturbed_f.shape)
edge_index_s_tensor = torch.tensor(edge_index_s, dtype=torch.int64)
edge_index_t_tensor = torch.tensor(edge_index_t, dtype=torch.int64)
label_tensor = torch.tensor(perturbed_l, dtype=torch.int64)
## graph_pair = GraphPair(edge_index_s, edge_index_t, fea_ori['arr_0'], perturbed_fe, perturbed_l) #, edge_index_s_tensor, edge_index_t_tensor)
graph_pair = GraphPair(edge_index_s_tensor, edge_index_t_tensor, x_s_tensor, x_t_tensor, label_tensor)



# model init
args.input_dim = perturbed_f.shape[1]
print('input_dim: ',args.input_dim)
print('edge_index_s_tensor: ',edge_index_s_tensor.shape)
print('edge_index_t_tensor: ',edge_index_t_tensor.shape)
print(args)
model = MFC_RMF(args.input_dim, args.hidden_dim, args.output_dim, args)
# print(summary(model))

# ground-truth
graph_edit_distance = nx.graph_edit_distance(G_o,perturbed_graph)
similarity = math.exp(-graph_edit_distance / ((G_o.number_of_nodes() + perturbed_graph.number_of_nodes()) / 2))
z_targe = torch.Tensor([similarity]) # 创建一个新对象（如张量）时，提供的数据应该是一个序列（sequence）



# model train
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
mse_loss = nn.MSELoss()
loss_history = []
for epoch in range(args.max_epoch):
    model.train()
    z, label, label_exp = model.forward(graph_pair)
    print('z: ',z[0].item())
    #print(type(z_targe), z_targe.shape, z_targe)
    loss = mse_loss(z[0], z_targe)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    print('epoch:{}, loss:{}'.format(epoch, loss))



# loss display
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(range(len(loss_history)), loss_history)
plt.ylabel('Loss')
plt.xlabel('Epoch : {}'.format(args.max_epoch))
plt.title('GraphPair_'+str(j)+ ' Training Loss: epoch' + str(args.max_epoch)+' lr'+ str(args.lr))
plt.text(0, loss_history[0], str(format(loss_history[0],'.8f')))
print(format(loss_history[(args.max_epoch - 1)],'.15f'))
plt.text(round(args.max_epoch/10*9) , loss_history[(args.max_epoch - 1)], str(format(loss_history[(args.max_epoch - 1)],'.10f')),
         horizontalalignment='left')


# time stamp
timestamp = time.time()
localtime = time.localtime(timestamp)
formatted_time = time.strftime('%Y%m%d_%H%M%S',localtime)

# Save
plt.savefig('./training_loss/GraphPair_'+str(j)+ '_n'+str(node)+'_d'+str(data_num)+'_Training_Loss_epoch_' + str(args.max_epoch) + '_lr_'+ str(args.lr)+'_'+str(formatted_time)+'.svg', format='svg')
plt.show()
torch.save(model, './model_save/model_GP_'+str(j)+ '_n_'+str(node)+'_d_'+str(data_num)+'_e_' + str(args.max_epoch) + '_l_'+ str(args.lr)+'_'+str(formatted_time)+'.pth')



# model reload
model.eval()
z_p = []
loss_history2 = []
for i in range(perturbed_a.shape[0]):
    print('GP：', i, '===============================')
    perturbed_graph_a2 = perturbed_a[i]
    # perturbed_graph_l2 = perturbed_l[i]
    perturbed_f2 = perturbed_fe[i]
    perturbed_graph2 = nx.Graph(perturbed_graph_a2)
    edge_index_t2 = np.array(list(perturbed_graph2.edges)).T

    # 转换成 PyTorch 张量
    # adj_tensor = torch.tensor(adj_ori['arr_0'], dtype=torch.int64)
    # adj_p_tensor = torch.tensor(perturbed_graph_a, dtype=torch.int64)
    #x_s_tensor = torch.tensor(fea_ori['arr_0'], dtype=torch.float32)
    x_t_tensor2 = torch.tensor(perturbed_f2, dtype=torch.float32)
    # print(fea_ori['arr_0'].shape)
    # print(perturbed_f.shape)
    # edge_index_s_tensor = torch.tensor(edge_index_s, dtype=torch.int64)
    edge_index_t_tensor2 = torch.tensor(edge_index_t2, dtype=torch.int64)
    # label_tensor = torch.tensor(perturbed_l, dtype=torch.int64)
    ## graph_pair = GraphPair(edge_index_s, edge_index_t, fea_ori['arr_0'], perturbed_fe, perturbed_l) #, edge_index_s_tensor, edge_index_t_tensor)
    graph_pair2 = GraphPair(edge_index_s_tensor, edge_index_t_tensor2, x_s_tensor, x_t_tensor2, label_tensor)

    # ground-truth
    graph_edit_distance2 = nx.graph_edit_distance(G_o, perturbed_graph2)
    similarity2 = math.exp(-graph_edit_distance2 / ((G_o.number_of_nodes() + perturbed_graph2.number_of_nodes()) / 2))
    z_targe2 = torch.Tensor([similarity2])  # 创建一个新对象（如张量）时，提供的数据应该是一个序列（sequence）

    with torch.no_grad():
         z_prediction, label2, label_exp2 = model(graph_pair2)  ###
         loss2 = mse_loss(z_prediction[0], z_targe2)
         loss_history2.append(loss2.item())
         z_p.append(z_prediction[0].item())
         print(' loss: '+ str(loss2.item()))
         print(' score: ' + str(z_prediction[0].item()))

# print(z_p)
# print(type(z_p))
np.savetxt('./similarity score/z'+ 'GP_'+str(j)+'_node_'+str(node)+'_data_'+str(data_num)+'model_'+str(formatted_time)+'.txt', z_p)
np.savetxt('./prediction_loss/GraphPair_' + str(j) + '_n' + str(node) + '_d' + str(data_num) + '_Prediction_Loss_epoch_' + str(
        args.max_epoch) + '_lr_' + str(args.lr) + '_' + str(formatted_time) +'.txt', loss_history2)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(range(len(loss_history2)), loss_history2)
plt.ylabel('Loss')
plt.xlabel('GraphPair_ : {}'.format(perturbed_a.shape[0]))
plt.title('Prediction Loss_' + 'GP_'+str(j)+ '_n_'+str(node)+'_d_'+str(data_num)+'_e_' + str(args.max_epoch) + '_l_'+ str(args.lr))
#plt.text(0, loss_history2[0], str(format(loss_history2[0], '.8f')))
#print(format(loss_history2[(args.max_epoch - 1)], '.15f'))
# plt.text(round(args.max_epoch / 10 * 9), loss_history2[(args.max_epoch - 1)],
         #str(format(loss_history2[(args.max_epoch - 1)], '.10f')),
        # horizontalalignment='left')
plt.savefig('./prediction_loss/GraphPair_' + str(j) + '_n' + str(node) + '_d' + str(data_num) + '_Prediction_Loss_epoch_' + str(
        args.max_epoch) + '_lr_' + str(args.lr) + '_' + str(formatted_time) + '.svg', format='svg')
plt.show()






