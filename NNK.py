import numpy as np
# from Non_neg_qpsolver import non_negative_qpsolver
from Modules.Construction.Non_neg_qpsolver import non_negative_qpsolver
import scipy.sparse as sparse
import Modules.Construction.utils as utils

def K_nearest_neighbors(X, metric, param, p):
    """
    Function to generate similarity matrix and mask by cosine or rbf
    :param X: Database matrix
    :param metric: Similarity metric to use for finding neighbors: cosine, rbf
    :param param: number of neighbors to use for NNK
    :param p: type of Lp distance to use (if used)
    :return K: Similarity matrix
    :return mask: each row corresponds to the neighbors to be considered for NNK optimization
    """
    if metric == 'cosine':
        # print("np.shape(X):", np.shape(X))
        # print("np.linalg.norm(X, axis=1):", np.linalg.norm(X, axis=1))
        # print("np.linalg.norm(X, axis=1)[:, None]:", np.linalg.norm(X, axis=1)[:, None])
        print("np.linalg.norm(X, axis=1)[:, None].shape:", np.linalg.norm(X, axis=1)[:, None].shape)
        X_normalized = X / np.linalg.norm(X, axis=1)[:, None]  # [:, None]改变序列
        print("X_normalized:", X_normalized)
        K = 0.5 + np.dot(X_normalized, X_normalized.T) / 2.0      #####???????
        mask = utils.create_directed_mask(D=K, param=param, D_type='similarity')
        print("K:", K)
        print("K.shape:", K.shape)
    elif metric == 'rbf':
        D = utils.create_distance_matrix(X=X, p=p)
        mask = utils.create_directed_mask(D=D, param=param, D_type='distance')

        # sigma = np.std(D[:, mask[:, -1]])
        # sigma = np.mean(D[:, mask[:, -1]]) / 3
        sigma = 1
        K = np.exp(-(D ** 2) / (2 * sigma ** 2))
        #print(K)
        #print(K.shape)
    # 这个马氏距离是我自己写的，感觉有点奇怪
    elif metric == 'mahalanobis':
        K = utils.create_distance_mahalanobis(X)
        mask = utils.create_directed_mask(D=K, param=param, D_type='distance')
    else:
        raise Exception("Unknown metric: " + metric)

    return K, mask


def nnk_graph(K, mask, param, reg=1e-6):
    """
    Function to generate NNK graph given similarity matrix and mask
    :param K: Similarity matrix
            相似度矩阵（就是输入核K）
    :param mask: each row corresponds to the neighbors to be considered for NNK optimization
            每行对应于 NNK 优化要考虑的邻居
    :param param: maximum number of neighbors for each node
            每个节点的最大邻居数
    :param reg: weights below this threshold are removed (set to 0)
            删除低于此阈值的权重（设置为 0）
    :return: Adjacency matrix of size num of nodes x num of nodes
            节点数 x 节点数的邻接矩阵
    """
    # 节点数
    num_of_nodes = K.shape[0]
    # 邻居初始索引
    neighbor_indices = np.zeros((num_of_nodes, param))
    # 权重初始值
    weight_values = np.zeros((num_of_nodes, param))
    # 误差初始值
    error_values = np.zeros((num_of_nodes, param))

    # 算法里面的循环
    for node_i in range(num_of_nodes):
        # 最近邻索引
        non_zero_index = np.array(mask[node_i, :])
        # 从最近邻中删掉自己
        non_zero_index = np.delete(non_zero_index, np.where(non_zero_index == node_i))
        # K_i 非零索引的笛卡尔积映射，就是算法中的Phi_S
        K_i = K[np.ix_(non_zero_index, non_zero_index)]
        # 这一步就是phi_i
        k_i = K[non_zero_index, node_i]
        # 这一步就是在计算算法中的theta_S
        x_opt, check = non_negative_qpsolver(K_i, k_i, k_i, reg)
        # 误差值计算
        error_values[node_i, :] = K[node_i, node_i] - 2 * np.dot(x_opt, k_i) + np.dot(x_opt, np.dot(K_i, x_opt))
        # 权重矩阵计算
        weight_values[node_i, :] = x_opt
        # 邻居索引
        neighbor_indices[node_i, :] = non_zero_index
        # print(neighbor_indices)

    row_indices = np.expand_dims(np.arange(0, num_of_nodes), 1)
    row_indices = np.tile(row_indices, [1, param])
    # 邻接矩阵
    adjacency = sparse.coo_matrix((weight_values.ravel(), (row_indices.ravel(), neighbor_indices.ravel())),
                                  shape=(num_of_nodes, num_of_nodes))
    # 误差矩阵
    error = sparse.coo_matrix((error_values.ravel(), (row_indices.ravel(), neighbor_indices.ravel())),
                                  shape=(num_of_nodes, num_of_nodes))
    # Alternate way of doing: error_index = sparse.find(error > error.T); adjacency[error_index[0], error_index[
    # 1]] = 0
    # 输出最后的邻接矩阵
    adjacency = adjacency.multiply(error <= error.T)
    adjacency = adjacency.maximum(adjacency.T)
    return adjacency


def knn_graph(X, mask, knn_param, reg=1e-6):
    """
    Function to generate KNN graph given similarity matrix and mask
    :param X: Similarity matrix
    :param mask: each row corresponds to the neighbors to be considered for NNK optimization
    :param knn_param: maximum number of neighbors for each node
    :param reg: weights below this threshold are removed (set to 0)
    :return: Adjacency matrix of size num of nodes x num of nodes
    """
    num_of_nodes = X.shape[0]
    neighbor_indices = np.zeros((num_of_nodes, knn_param))
    weight_values = np.zeros((num_of_nodes, knn_param))
    for node_i in range(num_of_nodes):
        non_zero_index = np.array(mask[node_i, :])
        non_zero_index = np.delete(non_zero_index, np.where(non_zero_index == node_i))
        g_i = X[non_zero_index, node_i]
        g_i[g_i < reg] = 0
        weight_values[node_i, :] = g_i
        neighbor_indices[node_i, :] = non_zero_index
    row_indices = np.expand_dims(np.arange(0, num_of_nodes), 1)
    row_indices = np.tile(row_indices, [1, knn_param])
    adjacency = sparse.coo_matrix((weight_values.ravel(), (row_indices.ravel(), neighbor_indices.ravel())),
                                  shape=(num_of_nodes, num_of_nodes))
    # Symmetrize adjacency matrix
    adjacency = adjacency.maximum(adjacency.T)
    return adjacency
