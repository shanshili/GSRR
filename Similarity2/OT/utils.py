from sklearn.neighbors import kneighbors_graph
import networkx as nx
import numpy as np
import math


def lb_to_xy(lat, lon):
    r = 6371
    theta = np.pi/2 - np.radians(lat)
    phi = np.radians(lon)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    coo = np.vstack([x, y])
    return coo

def find_value_according_index_list(aim_list, index_list):
    i = 0
    reslut_list = []
    while i < len(index_list):
        reslut_list.append(aim_list[index_list[i]])
        i = i + 1
    return reslut_list




def get_h_hop_neighbors(G, node, hop=1):
    '''

    :param G: 图
    :param node: 中心节点
    :param hop: 跳数
    :return:
    '''
    output = {}
    layers = dict(nx.bfs_successors(G, source=node, depth_limit=hop))
    nodes = [node]
    for i in range(1, hop+1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x, []))
        nodes = output[i]
    return output, output[hop]

def communication_energy_loss(G, node, neighbors, E_0, E_elec=600, alpha=3, beta=120, bit=1000):
    # Neighbor_number = len(neighbors)
    Energy = E_0
    for i in neighbors:
        dis = nx.shortest_path_length(G, node, i)
        Energy -= (2 * E_elec + beta * dis ** alpha) * 1e-9 * bit
    return Energy


def network_life(pos, selected_nodes, k):
    pos_selected = find_value_according_index_list(pos, selected_nodes)
    # 构图
    A = kneighbors_graph(pos_selected, k)
    G = nx.Graph(A)
    rad = nx.radius(G)
    # 收集每轮剩余能量
    # energy_collect = []
    # 初始能量
    energy_loss = [6] * nx.number_of_nodes(G)
    # 通信轮次
    for i in range(100000):
        # 节点
        for j in range(nx.number_of_nodes(G)):
            # 找到通信范围
            neighbor_dict, neighbor_list = get_h_hop_neighbors(G, j, rad)
            # 计算剩余能量
            node_energy_loss = communication_energy_loss(G, j, neighbor_list, energy_loss[j])
            energy_loss[j] = node_energy_loss
            # print(j)
        # print(energy_loss)
        # energy_collect.append(energy_loss.copy())  # 使用浅拷贝避免覆盖

        if any(v <= 0 for v in energy_loss):  # 判断列表中是否有小于0的数
            rest_energy = [u for u in energy_loss if u > 0]     # 删除小于0的数
            res_energy_avg = sum(rest_energy) / len(selected_nodes)
            # print('网络寿命为：{}'.format(i + 1))
            break
    # energy_collect = np.array(energy_collect)
    return i + 1, res_energy_avg


