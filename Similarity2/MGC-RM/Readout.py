from model import readout, GraphAutoencoder, AutoEncoder
from torch import nn, optim
import torch
import argparse
import numpy as np
from GraphConstruct2 import topological_features_construct
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from utils import find_value_according_index_list
from GraphConstruct2 import location_graph


parser = argparse.ArgumentParser(
    description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
# parser.add_argument("--name", type=str, default="Citeseer")
parser.add_argument("--max_epoch", type=int, default=21)
parser.add_argument("--lr", type=float, default=0.0001)
#parser.add_argument("--n_clusters", default=6, type=int)
parser.add_argument("--hidden_dim", default=1000, type=int)
parser.add_argument("--output_dim", default=250, type=int)
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
location_file = np.load('./origin_data/location'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', allow_pickle=True)
fea_ori = np.load('./origin_data/fea_ori'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', allow_pickle=True)
A_o = adj_ori['arr_0']
G_o = nx.Graph(adj_ori['arr_0'])
location = location_file['arr_0']
fea_o = fea_ori['arr_0']
# model init
args.input_dim = node
print('input_dim: ',args.input_dim)


# 读取节点排序
node_list=[]
with open('WPR_result\_node_list_GP_75_node_450_data_2000.txt','r') as f:
    for line in f:
        node_list.append(int(list(line.strip('\n').split(','))[0]))

# 构造多维图数据集
selected_node = [[] for _ in range(len(node_list))]
x = [None] * (int(len(node_list)) + 1)
g = [[] for _ in range(len(node_list))]
A = [[] for _ in range(len(node_list))]
fea_list = [[] for _ in range(len(node_list))]
location_list = [[] for _ in range(len(node_list))]
i = 0 # 采样计数
for select_node in range(0,int(len(node_list))+1,1):
    x[i] = select_node+1 # 已选择节点数目
    selected_node[i] = node_list[:select_node+1]
    fea_list[i] = find_value_according_index_list(fea_o, selected_node[i])
    location_list[i] = find_value_according_index_list(location, selected_node[i])
    g[i], A[i] = location_graph(location_list[i])
    print(select_node+1)


GraphAutoencoder_ = GraphAutoencoder(input_size,hidden_size,output_size)
optimizer = optim.Adam(readout.parameters(), lr=1e-3)
loss_fun = nn.MSELoss()