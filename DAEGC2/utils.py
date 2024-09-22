import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import normalize

from torch_geometric.datasets import Planetoid


# def get_dataset(dataset):
#     datasets = Planetoid('./dataset', dataset)
#     return datasets
#
def data_preprocessing(A,location_g,x):  # 生成邻接矩阵？？对邻接矩阵归一化
    # A = np.array(nx.adjacency_matrix(location_g).todense())
    # print(A,A.data,A.shape)
    adj = torch.Tensor(A.toarray())
    # adj = torch.sparse_coo_tensor(
    #     A.Coords,
    #     torch.FloatTensor(A.data),
    #     torch.Size(A.shape)
    # ).to_dense()
    adj_label = adj

    adj = torch.eye(x.shape[0])
    adj = normalize(adj, norm="l1")
    adj = torch.from_numpy(adj).to(dtype=torch.float)
    #print('adj{},adj_label{}'.format(adj,adj_label))
    #print('adj{},adj_label{}'.format(np.shape(adj), np.shape(adj_label)))
    return adj,adj_label

def get_M(adj):
    # adj_numpy = adj.cpu().numpy()
    adj_numpy = adj
    # t_order
    t=2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)


