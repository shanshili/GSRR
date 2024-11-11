import networkx as nx
import numpy as np
import math
import numpy
from GraphConstruct2 import topological_features_construct
from model import AutoEncoder
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


def find_value_according_index_list(aim_list, index_list):  # 索引转换 从排序索引找对应节点
    i = 0
    reslut_list = []
    while i < len(index_list):
        reslut_list.append(aim_list[index_list[i]])
        i = i + 1
    return reslut_list

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
    layers = dict(nx.bfs_successors(G, source=node, depth_limit=hop))  # 广度优先搜索（BFS）
    nodes = [node]
    for i in range(1, hop+1):  # 遍历每一跳
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x, []))
        nodes = output[i]
    return output, output[hop]


def communication_energy_loss(G, node, neighbors, E_0, E_elec=600, alpha=3, beta=120, bit=1000):
    # Neighbor_number = len(neighbors)
    Energy = E_0
    for i in neighbors:
        dis = nx.shortest_path_length(G, node, i)  # 和邻居之间的最短路径
        Energy -= (2 * E_elec + beta * dis ** alpha) * 1e-9 * bit   # 和范围半径内的每个邻居通信
    return Energy

def network_life(G):
    n = G.number_of_nodes()
    # 初始能量
    energy_loss = [20] * nx.number_of_nodes(G)
    res_energy_avg = 0
    i = 0
    """
    如果图不是连通的，nx.radius 函数会抛出 NetworkXError，因为半径的定义要求图必须是连通的。
    对于非连通图，你需要分别计算每个连通子图的半径。
    """
    try: # 连通图
        # 尝试计算整个图的半径
        radius = nx.radius(G)
        # 通信轮次
        for i in range(100000):
            # 节点
            for j in range(n):
                # 找到通信范围
                neighbor_dict, neighbor_list = get_h_hop_neighbors(G, j, radius)
                # 计算剩余能量
                node_energy_loss = communication_energy_loss(G, j, neighbor_list, energy_loss[j])
                energy_loss[j] = node_energy_loss
                # print(j)
            # print(energy_loss)
            # energy_collect.append(energy_loss.copy())  # 使用浅拷贝避免覆盖

            if any(v <= 0 for v in energy_loss):  # 判断列表中是否有小于0的数
                rest_energy = [u for u in energy_loss if u > 0]  # 删除小于0的数
                res_energy_avg = sum(rest_energy) / n
                # print('网络寿命为：{}'.format(i + 1))
                break
        # energy_collect = np.array(energy_collect)
        print('i',i)

    except nx.NetworkXError as e:  # 非连通图
        # 获取所有连通子图
        connected_components = list(nx.connected_components(G))
        # 计算连通子图的个数
        num_connected_components = len(connected_components)
        print('connected: ',n)
        print('connected_components: ', num_connected_components)
        # 计算每个连通子图的半径
        radii = []
        ii = [[] for _ in range(num_connected_components)]
        res_energy_avg_sub = [[] for _ in range(num_connected_components)]
        com = 0
        for cc in connected_components:
            # print(cc)
            subgraph = G.subgraph(cc)
            subgraph_radius = nx.radius(subgraph)
            radii.append(subgraph_radius)
            # 通信轮次
            for x in range(100000):
                # 节点
                for j in range(n):
                    # 找到通信范围
                    neighbor_dict_sub, neighbor_list_sub = get_h_hop_neighbors(G, j, subgraph_radius)
                    # 计算剩余能量
                    node_energy_loss_sub = communication_energy_loss(G, j, neighbor_list_sub, energy_loss[j])
                    energy_loss[j] = node_energy_loss_sub
                    # print(j)
                # print(energy_loss)
                # energy_collect.append(energy_loss.copy())  # 使用浅拷贝避免覆盖

                if any(v <= 0 for v in energy_loss):  # 判断列表中是否有小于0的数
                    rest_energy = [u for u in energy_loss if u > 0]  # 删除小于0的数
                    res_energy_avg_sub[com] = sum(rest_energy) / n
                    # print('网络寿命为：{}'.format(i + 1))
                    break

            ii[com] = x + 1
            com = com+1

            # print('ii',ii)
            # print('res_energy_avg_sub',res_energy_avg_sub)
        i = max(ii)
        res_energy_avg = sum(res_energy_avg_sub)/len(res_energy_avg_sub)
        print(ii)
        print(i)
        print(res_energy_avg_sub)
        print(res_energy_avg)
            # energy_collect = np.array(energy_collect)
    # print('i',i + 1)
    # ('res_energy_avg_sub', res_energy_avg)

    return i + 1, res_energy_avg

def MSE_node_feature(g,node):
    topological_features = topological_features_construct(g)
    data = np.array(topological_features)
    norm_scalar = MinMaxScaler()
    data = np.transpose(norm_scalar.fit_transform(data))
    data = torch.tensor(data, dtype=torch.float32)
    model_path = './model_save/autoencoder.pth'
    autoencoder = AutoEncoder()
    autoencoder = torch.load(model_path)
    autoencoder.eval()
    loss_fun = nn.MSELoss()
    loss_history = []
    first_node_encode = []
    with torch.no_grad():
        encoded, decoded = autoencoder(data)
        loss = loss_fun(decoded, data)
        loss_history.append(loss.item())
        # print(encoded)
        # print(loss)
        first_node_encode.append(encoded[node].item())
        # print(' loss: ' + str(loss.item()))
        # print(' encode: ' + str(first_node_encode))
    return first_node_encode

def mean_squared_error(y_true, y_pred):
    """
    计算均方误差 (MSE)
    参数:
    y_true (array-like): 真实值
    y_pred (array-like): 预测值
    返回:
    float: 均方误差
    """
    # 确保输入是 NumPy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # 计算误差平方
    squared_errors = (y_true - y_pred) ** 2
    # 计算均方误差
    mse = np.mean(squared_errors)
    return mse



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

def DS(lon, lat, num_selected):
    """
    原始DS，归一化坐标
    Rst = (2 / (math.pi * num_selected)) ** 0.5
    g_distance，f_distance计算空洞半径
    不包含前三个点，整体数值较小，运算较慢
    """
    # 区域压缩(最大最小值)
    lon_max = np.max(lon)
    lon_min = np.min(lon)
    lat_max = np.max(lat)
    lat_min = np.min(lat)
    lon_n = (lon - lon_min) / (lon_max - lon_min)
    lat_n = (lat - lat_min) / (lat_max - lat_min)

    position = map(list, zip(lon_n, lat_n))  # 平面的，为了画图
    pos = list(position)
    # print(pos)

    hole_radius_f = []
    hole_radius_g = []
    for i in range(num_selected):
        for j in range(num_selected):
            if i == j: continue
            else:
                position_fi, position_fj, center_f, f_d = f_distance(pos[i], pos[j])
                position_gi, position_gj, center_g, g_d = g_distance(pos[i], pos[j])

            for k in range(num_selected):
                position_fmi, position_fmj, center_fm, fm_d = f_distance(center_f, pos[k])
                position_gmi, position_gmj, center_gm, gm_d = g_distance(center_g, pos[k])

                if fm_d >= f_d / 2 and gm_d >= f_d / 2: hole_radius_f.append(f_d / 2)
                if fm_d >= g_d / 2 and gm_d >= g_d / 2: hole_radius_g.append(g_d / 2)

    # print(hole_radius_f,hole_radius_g)
    hole_radius = hole_radius_f + hole_radius_g
    # print(hole_radius)
    r_min = min(hole_radius)
    r_max = max(hole_radius)
    # D = r_min / r_max

    Rst = (2 / (math.pi * num_selected)) ** 0.5

    return Rst / r_max




def find_holes(graph):
    # 使用 BFS 查找连通分量
    holes = list(nx.connected_components(graph))
    return holes

def compute_radius(hole, graph):
    # 计算空洞的半径
    max_radius = 0
    hole_graph = graph.subgraph(hole)
    eccentricities = nx.eccentricity(hole_graph)
    """
    使用 nx.eccentricity 函数来计算每个节点的偏心率（eccentricity），
    即从该节点到图中其他节点的最短路径的最大值。
    半径是所有偏心率中的最小值
    """
    radius = min(eccentricities.values())
    return radius

def compute_radius2(hole, positions):
    """
    考虑物理位置
    """
    # 计算空洞的半径
    # 提取空洞中的节点位置
    # print(hole)
    # print(positions)
    hole_positions = [positions[node] for node in hole]
    # 计算所有节点之间的距离
    if len(hole_positions) < 2:
        return 0  # 如果空洞中只有一个节点，半径为0
    distances = pdist(hole_positions)
    # 找到最大距离
    max_distance = np.max(distances)
    # 最大空洞半径是最大距离的一半
    max_radius = max_distance / 2
    return max_radius

def maximum_hole_radius(graph):
    """
    不考虑物理距离
    """
    holes = find_holes(graph)
    max_radius = 0
    for hole in holes:
        radius = compute_radius(hole, graph)
        max_radius = max(max_radius, radius)
    return max_radius

def maximum_hole_radius2(positions):
    """
    重新构图
    考虑物理距离，但是没有限制连接数量
    """
    # 构建图
    num_nodes = len(positions)
    Gn = nx.Graph()
    Gn.add_nodes_from(range(num_nodes))
    # 计算所有节点之间的距离
    distance_matrix = squareform(pdist(positions))
    # print(distance_matrix)
    # 添加边，如果两个节点之间的距离小于阈值，则认为它们是连通的
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if distance_matrix[i, j] < 1:
                Gn.add_edge(i, j)
    # print(Gn)
    # plt.figure()
    # nx.draw(Gn,pos = positions,with_labels=True, alpha = 0.4, node_size=10, font_size = 5)
    # plt.show()
    # 查找连通分量
    holes = find_holes(Gn)

    # 计算每个空洞的最大半径
    max_radius = 0
    for hole in holes:
        radius = compute_radius2(hole, positions)
        max_radius = max(max_radius, radius)

    return max_radius

def maximum_hole_radius3(positions, k=4):
    """
    重新构图
    考虑物理距离，考虑连接的邻居个数
    但是起始必须大于4个节点
    """
    # 构建图
    num_nodes = len(positions)
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    if num_nodes < 5:
        # 计算所有节点之间的距离
        distance_matrix = squareform(pdist(positions))
        # print(distance_matrix)
        # 添加边，如果两个节点之间的距离小于阈值，则认为它们是连通的
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if distance_matrix[i, j] < 1:
                    G.add_edge(i, j)
        # plt.figure()
        # nx.draw(G,pos = positions,with_labels=True, alpha = 0.4, node_size=10, font_size = 5)
        # plt.show()
    else:
        # 使用 cKDTree 找到每个节点的最近 k 个邻居
        tree = cKDTree(positions)
        _, indices = tree.query(positions, k=k + 1)  # k+1 是因为最近的一个点是自己
        # print(indices)
        # 添加边
        for i in range(num_nodes):
            for j in indices[i][1:]:  # 跳过第一个元素（即自身）
                G.add_edge(i, j)
    # 查找连通分量
    holes = find_holes(G)

    # 计算每个空洞的最大半径
    max_radius = 0
    for hole in holes:
        radius = compute_radius2(hole, positions)
        max_radius = max(max_radius, radius)

    return max_radius

def DS3(position,num_selected):
    """
    max_radius = maximum_hole_radius2(g) 或 maximum_hole_radius3(g)
    gamma_value = np.math.gamma((m / 2) + 1)
    denominator = ((gamma_value / np.pi ** (m / 2)) * (1 / n)) ** (1 / m)
    """
    m = 2
    n = num_selected
    max_radius = maximum_hole_radius2(position)
    # max_radius = maximum_hole_radius3(position)
    gamma_value = np.math.gamma((m / 2) + 1)
    denominator = ((gamma_value / np.pi ** (m / 2)) * (1 / n)) ** (1 / m)
    print('max_radius',max_radius)
    print('Rst',denominator)
    if max_radius > 0:
        DS = denominator / max_radius
    else:
        DS = denominator / 0.5
    return DS

def DS2(g,num_selected):
    """
    max_radius = maximum_hole_radius(g)
    gamma_value = np.math.gamma((m / 2) + 1)
    denominator = ((gamma_value / np.pi ** (m / 2)) * (1 / n)) ** (1 / m)
    """
    m = 2
    n = num_selected
    max_radius = maximum_hole_radius(g)
    gamma_value = np.math.gamma((m / 2) + 1)
    denominator = ((gamma_value / np.pi ** (m / 2)) * (1 / n)) ** (1 / m)
    print('max_radius',max_radius)
    print('Rst',denominator)
    if max_radius > 0:
        DS = denominator / max_radius
    else:
        DS = denominator / 0.01
    return DS