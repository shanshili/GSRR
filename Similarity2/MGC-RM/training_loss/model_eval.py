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

model = torch.load('resnet.pth')
# model reload
model.eval()
z_p = []
loss_history2 = []
for i in range(perturbed_a.shape[0]):
    print('======= perturbed：', i, '========================')
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
         print('GP '+ str(i) + ' loss:'+ str(loss2.item()))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(range(len(loss_history2)), loss_history2)
plt.ylabel('Loss')
plt.xlabel('GraphPair_ : {}'.format(perturbed_a.shape[0]))
plt.title('Similarity Loss_' + 'GP_'+str(i)+ '_n_'+str(node)+'_d_'+str(data_num)+'_e_' + str(args.max_epoch) + '_l_'+ str(args.lr))
#plt.text(0, loss_history2[0], str(format(loss_history2[0], '.8f')))
#print(format(loss_history2[(args.max_epoch - 1)], '.15f'))
# plt.text(round(args.max_epoch / 10 * 9), loss_history2[(args.max_epoch - 1)],
         #str(format(loss_history2[(args.max_epoch - 1)], '.10f')),
        # horizontalalignment='left')
plt.savefig('./prediction_loss/GraphPair_' + str(i) + '_n' + str(node) + '_d' + str(data_num) + '_Training_Loss_epoch_' + str(
        args.max_epoch) + '_lr_' + str(args.lr) + '_' + str(formatted_time) + '.svg', format='svg')
plt.show()



# print(z_p)
# print(type(z_p))
np.savetxt('./similarity score/z'+'_node_'+str(node)+'_data_'+str(data_num)+'model_'+str(formatted_time)+'.txt', z_p)


