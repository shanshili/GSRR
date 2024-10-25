import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.bipartite.projection import projected_graph

from GraphConstruct2 import location_graph
from dataprocess2 import get_data2

import os
# import time

# time_start=time.time()

def Read_data_CSV(dataset_in_project_dir):
    # 读取基准图数据 并打包为文件
    project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    Path1 = project_dir + dataset_in_project_dir
    tem_data, data_location = get_data2(Path1)  # 温度数据作为特征
    return tem_data, data_location

def memory_over(tem_data, data_location, node , data_num ):
    tem = tem_data[:node,:data_num]
    location = data_location[:node,:]
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
    return fea_ori, location

# def creat_edge_index()

def Construct_base_graph(node_location):
    # 构造基准图
    G,adj = location_graph(node_location)
    # print("G_0.nodes:",location_g.nodes)
    print("G_0.number_of_nodes:",G.number_of_nodes())    # 节点数：
    print("G_0.number_of_edges:",G.number_of_edges())    # 边数：

    return  G, adj


def Construct_perturbation_graph(G_0,fea_ori):
    # 对所有节点执行
    # perturbed_edge_set = []
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
        if (nx.is_isolate(G, i) == True):
            # edge_index = G.edges
            # perturbed_edge_set.append(edge_index)
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

    # print("perturbed_edge_set", len(perturbed_edge_set))
    # print(perturbed_adj_set)
    print("perturbed_adj_set", len(perturbed_adj_set))
    # print(perturbed_graph_label)
    print("perturbed_graph_label", len(perturbed_graph_label))
    # print(perturbed_graph_fea)
    print("perturbed_graph_fea", len(perturbed_graph_fea))

    return  perturbed_adj_set, perturbed_graph_label, perturbed_graph_fea


dataset_in_project_dir = '/dataset/北京-天津气象2021'
node = 754
data_num = 4000
tem_data, data_location = Read_data_CSV(dataset_in_project_dir)
fea_ori, location = memory_over(tem_data, data_location, node , data_num)
G_0, adj_ori = Construct_base_graph(location)
perturbed_adj_set, perturbed_graph_label, perturbed_graph_fea = Construct_perturbation_graph(G_0,fea_ori)



# 保存到文件
np.savez('./origin_data/adj_ori'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', adj_ori)
np.savez('./origin_data/fea_ori'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', fea_ori)
np.savez('./origin_data/location'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', location)
# np.savez('./perturbation_data/perturbed_edge_set.npz', perturbed_edge_set)  #
np.savez('./perturbation_data/perturbed_graph'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', perturbed_adj_set)  # 里面是一个(node, node, node)的张量：node个邻接矩阵
np.savez('./perturbation_data/perturbed_label'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', perturbed_graph_label)  # node
np.savez('./perturbation_data/perturbed_fea'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', perturbed_graph_fea)  # (node,node,data_num)
# np.savetxt("./perturbation_data/perturbed_graph.csv", perturbed_adj_set, delimiter=',')
# np.savetxt('./perturbation_data/perturbed_label.csv', perturbed_graph_label,delimiter=',')
# np.savetxt('./perturbation_data/perturbed_fea.csv', perturbed_graph_fea, delimiter=',')
# '''
# time_end=time.time()
# print('time cost',time_end-time_start,'s')