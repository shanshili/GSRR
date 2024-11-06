# import EntropyHub as EH
import networkx as nx
import numpy as np
import math
from sklearn.neighbors import kneighbors_graph


# update index_list


def find_value_according_index_list(aim_list, index_list):  # 索引转换 从排序索引找对应节点
    i = 0
    reslut_list = []
    while i < len(index_list):
        reslut_list.append(aim_list[index_list[i]])
        i = i + 1
    return reslut_list


# 时间序列的三种熵
# 近似熵
# def ApEn(Datalist, r=0.2, m=2):
#     th = r * np.std(Datalist)
#     return EH.ApEn(Datalist, m, r=th)[0][-1]
# # 样本熵
# def SampleEntropy2(Datalist, r, m=2):
#     th = r * np.std(Datalist) #容限阈值
#     return EH.SampEn(Datalist, m, r=th)[0][-1]
# # 模糊熵
# def FuzzyEn2(s:np.ndarray, r=0.2, m=2, n=2):
#     th = r * np.std(s)
#     return EH.FuzzEn(s, 2, r=(th, n))[0][-1]


def natural_connectivity(G):
    adj_spec = nx.adjacency_spectrum(G)
    adj_spec_exp = np.exp(adj_spec)
    adj_spec_exp_sum = np.sum(adj_spec_exp)
    n_c = np.log(adj_spec_exp_sum / adj_spec.shape)
    return n_c


def natural_connectivity2(G):
    # 构建拉普拉斯矩阵
    L = nx.laplacian_matrix(G).toarray()
    # 计算拉普拉斯矩阵的特征值
    eigenvalues = np.linalg.eigvalsh(L)
    # 去掉一个零特征值
    non_zero_eigenvalues = eigenvalues[eigenvalues > 1e-10]
    # 计算自然连通性
    n = G.number_of_nodes()
    # n = 1
    # log_sum = np.log(np.prod(non_zero_eigenvalues))
    # print(np.prod(non_zero_eigenvalues))
    log_sum = np.sum(np.log(non_zero_eigenvalues))
    natural_conn = log_sum / n
    return natural_conn

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


def f_distance(pos_i, pos_j):


    # f坐标(k=1:横坐标; k=2:纵坐标)
    position_fi = []
    position_fj = []
    center_f = []

    for k in range(2):
        if abs(pos_i[k] - pos_j[k]) >= 0.5:
            pos_fi = pos_i[k]
            pos_fj = pos_j[k]
        elif abs(pos_i[k] - pos_j[k]) < 0.5 and pos_i[k] >= pos_j[k]:
            pos_fi = pos_i[k]
            pos_fj = pos_j[k] + 1
        elif abs(pos_i[k] - pos_j[k]) < 0.5 and pos_i[k] < pos_j[k]:
            pos_fi = pos_i[k] + 1
            pos_fj = pos_j[k]
        position_fi.append(pos_fi)  # 第i个节点的坐标
        position_fj.append(pos_fj)  # 第j个节点的坐标

        # f中心(横坐标)
        if (position_fi[k] + position_fj[k]) / 2 >= 1:
            m_f = (position_fi[k] + position_fj[k]) / 2 - 1
        else:
            m_f = (position_fi[k] + position_fj[k]) / 2
        center_f.append(m_f)  # f中心的坐标

    # f距离
    f_d = math.sqrt((position_fi[0] - position_fj[0]) ** 2 + (position_fi[1] - position_fj[1]) ** 2)

    return position_fi, position_fj, center_f, f_d

def g_distance(pos_i, pos_j):


    # g坐标(k=1:横坐标; k=2:纵坐标)
    position_gi = []
    position_gj = []
    center_g = []

    for k in range(2):
        if abs(pos_i[k] - pos_j[k]) <= 0.5:
            pos_gi = pos_i[k]
            pos_gj = pos_j[k]
        elif abs(pos_i[k] - pos_j[k]) > 0.5 and pos_i[k] >= pos_j[k]:
            pos_gi = pos_i[k]
            pos_gj = pos_j[k] + 1
        elif abs(pos_i[k] - pos_j[k]) > 0.5 and pos_i[k] < pos_j[k]:
            pos_gi = pos_i[k] + 1
            pos_gj = pos_j[k]
        position_gi.append(pos_gi)  # 第i个节点的坐标
        position_gj.append(pos_gj)  # 第j个节点的坐标

        # g中心(横坐标)
        if (position_gi[k] + position_gj[k]) / 2 >= 1:
            m_g = (position_gi[k] + position_gj[k]) / 2 - 1
        else:
            m_g = (position_gi[k] + position_gj[k]) / 2
        center_g.append(m_g)  # f中心的坐标

    # g距离
    g_d = math.sqrt((position_gi[0] - position_gj[0]) ** 2 + (position_gi[1] - position_gj[1]) ** 2)

    return position_gi, position_gj, center_g, g_d


def f_distance_2(lon, lat, i, j):
    # lon: 横坐标列表
    # lat: 纵坐标列表
    # i: 节点i
    # j: 节点j

    # f坐标(横坐标)
    if abs(lon[i] - lon[j]) >= 0.5:
        lon_fi = lon[i]
        lon_fj = lon[j]
    elif abs(lon[i] - lon[j]) < 0.5 and lon[i] >= lon[j]:
        lon_fi = lon[i]
        lon_fj = lon[j] + 1
    elif abs(lon[i] - lon[j]) < 0.5 and lon[i] < lon[j]:
        lon_fi = lon[i] + 1
        lon_fj = lon[j]

    # f坐标(纵坐标)
    if abs(lat[i] - lat[j]) >= 0.5:
        lat_fi = lat[i]
        lat_fj = lat[j]
    elif abs(lat[i] - lat[j]) < 0.5 and lat[i] >= lat[j]:
        lat_fi = lat[i]
        lat_fj = lat[j] + 1
    elif abs(lat[i] - lat[j]) < 0.5 and lat[i] < lat[j]:
        lat_fi = lat[i] + 1
        lat_fj = lat[j]

    # f距离
    f_d = math.sqrt((lon_fi - lon_fj) ** 2 + (lat_fi - lat_fj) ** 2)

    # f中心(横坐标)
    if (lon_fi + lon_fj) / 2 >= 1: m_fi = (lon_fi + lon_fj) / 2 - 1
    else: m_fi = (lon_fi + lon_fj) / 2

    return lon_fi, lat_fi, lon_fj, lat_fj, f_d

def g_distance_2(lon, lat, i, j):
    # lon: 横坐标列表
    # lat: 纵坐标列表
    # i: 节点i
    # j: 节点j

    # g坐标(横坐标)
    if abs(lon[i] - lon[j]) <= 0.5:
        lon_gi = lon[i]
        lon_gj = lon[j]
    elif abs(lon[i] - lon[j]) > 0.5 and lon[i] >= lon[j]:
        lon_gi = lon[i]
        lon_gj = lon[j] + 1
    elif abs(lon[i] - lon[j]) > 0.5 and lon[i] < lon[j]:
        lon_gi = lon[i] + 1
        lon_gj = lon[j]

    # g坐标(纵坐标)
    if abs(lat[i] - lat[j]) <= 0.5:
        lat_gi = lat[i]
        lat_gj = lat[j]
    elif abs(lat[i] - lat[j]) > 0.5 and lat[i] >= lat[j]:
        lat_gi = lat[i]
        lat_gj = lat[j] + 1
    elif abs(lat[i] - lat[j]) > 0.5 and lat[i] < lat[j]:
        lat_gi = lat[i] + 1
        lat_gj = lat[j]

    # f距离
    g_d = math.sqrt((lon_gi - lon_gj) ** 2 + (lat_gi - lat_gj) ** 2)

    return lon_gi, lat_gi, lon_gj, lat_gj, g_d


def lb_to_xy(lat, lon):
    r = 6371
    theta = np.pi/2 - np.radians(lat)
    phi = np.radians(lon)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    coo = np.vstack([x, y])
    return coo