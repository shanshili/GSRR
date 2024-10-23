import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.bipartite.projection import projected_graph

from GraphConstruct2 import location_graph
from dataprocess2 import get_data2

import os

"""
# 读取原始特征
fea_ori = pd.read_csv('CIMIS/raw_data/dataset.csv').values
# 读取基准图结构
adj_ori = pd.read_csv('CIMIS/raw_data/adjacency_matain.csv', index_col=0).values
"""

# 读取基准图数据
project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
Path1 = project_dir + '/dataset/北京-天津气象2021'
tem_data, data_location = get_data2(Path1)  # 温度数据作为特征
tem = tem_data[:450,:2000]
location = data_location[:450,:]

# # 创建内存映射文件
# filename = 'tem_data.memmap'
# shape = (754, 1500)# (754,8760)#
# dtype = np.float32
# # 创建一个空的内存映射文件
# large_array = np.memmap(filename, mode='w+', shape=shape, dtype=dtype)
# # 填充数据（如果需要）
# large_array[:] = tem.astype(np.float32)
# # 关闭内存映射文件
# large_array.flush()
# del large_array

# 重新打开内存映射文件
# large_array = np.memmap(filename, mode='r', shape=shape, dtype=dtype)
fea_ori = tem.astype(np.float32)# large_array # tem.astype(np.float32)
# print("fea_ori:",fea_ori)

# 构造基准图
G_0,adj_ori = location_graph(location)
# print("G_0.nodes:",location_g.nodes)
print("G_0.number_of_nodes:",G_0.number_of_nodes())    # 节点数：
print("G_0.number_of_edges:",G_0.number_of_edges())    # 边数：

# 对所有节点执行
perturbed_adj_set = []
perturbed_graph_label = []
perturbed_graph_fea = []
for i in range(G_0.number_of_nodes()):  # 450 node

    # 找到与中心节点相邻的所有边
    edge_list_0 = G_0.edges(i)
    # print("edge_list_0:\n",edge_list_0)


    # 需要掩蔽的特征位置(中心节点及其一阶邻居)
    # fea_mask_list = list(nx.all_neighbors(G_0, i))
    # # print(fea_mask_list)
    # fea_mask_list.append(i)
    # print(fea_mask_list)


    # copy基准图（防止直接删了基准图上的边）
    G = G_0.copy()
    # 删除与中心节点相邻的所有边
    G.remove_edges_from(edge_list_0)


    # copy原始特征
    # large_array = np.memmap(filename, mode='r', shape=shape, dtype=dtype)
    # fea = large_array.copy()
    # large_array.flush()
    # del large_array
    fea = fea_ori.copy()


    # 判断中心节点是否孤立
    if (nx.is_isolate(G, i) ==
            True):
        adj_perturbed = nx.to_pandas_adjacency(G)   # 已孤立，则汇总起来邻接矩阵
        perturbed_adj_set.append(adj_perturbed)
        # 获取标签 节点度/总节点数 （重要性特征？？）
        # print("G_0.degree[i]",G_0.degree[i])
        # print("G_0.number_of_nodes()", G_0.number_of_nodes())
        perturbed_graph_label.append(G_0.degree[i] / G_0.number_of_nodes())

        # 掩蔽特征
        fea[i] = 1e-20
        perturbed_graph_fea.append(fea) # 掩蔽点特征为0

    else: print('node {} is not isolated'.format(i))



# print(perturbed_adj_set)
print("perturbed_adj_set",len(perturbed_adj_set))
# print(perturbed_graph_label)
print("perturbed_graph_label",len(perturbed_graph_label))
# print(perturbed_graph_fea)
print("perturbed_graph_fea",len(perturbed_graph_fea))


# 保存到文件
np.savez('./perturbation_data/perturbed_graph.npz', perturbed_adj_set)  # 里面是一个(n, 450, 450)的张量：450个邻接矩阵
np.savez('./perturbation_data/perturbed_label.npz', perturbed_graph_label)  # 450
np.savez('./perturbation_data/perturbed_fea.npz', perturbed_graph_fea)  # 450
np.savetxt("./perturbation_data/perturbed_graph.csv", perturbed_adj_set, delimiter=',')
np.savetxt('./perturbation_data/perturbed_label.csv', perturbed_graph_label,delimiter=',')  # 450
np.savetxt('./perturbation_data/perturbed_fea.csv', perturbed_graph_fea, delimiter=',')  # 450
# '''
