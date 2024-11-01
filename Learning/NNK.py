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
    row_i�C���Ϥ�F�e��`q�em�,繐8P�!��{�b�ʚ=�..3h~(m�R�0�i�ě��
��.&1�$$�D�!V~�}[wx���'v�Ih�O�#�g�{ñ[�a�:���m�(�t����p�@�.�
��n����^d�"�p����n�:ӟ�Í��f�Ti���l9dܸ�t�����_<�H:̫�Y��;��5͟&h8��U��>r�tg
7L^��g��H�.�p �b�7��f����G.g����1|�@o��o3}u���&=D�q?���jU��H�����o� �ܲ�����G��6�_���@q��j��us_���k����#ڙw�-�zm�ϧ��$��U�Dc�K�ne-J0����1��>���gFU��r�>1i>|��j1��]��c�B���6v~�����L	Lq���F\"�!���2Ɔ6* �(�3�1�ɀ(%|G��h��ت��GNuwZ�P�pf����zn?1򡼣�6?1� �
_�ӭ`8Cy��b�F����g��G-�ԌZ� ����~e'L�3|Ʒ�Њ;$��c�h�W�����\�~�y��] �0�w��J�t4?�^Iwo*�ˎ�w�V��p�����\�l���R�Hߘzs�y�˝�!����&\G}�/W�fh�ג"�w�*N�,���@�����I/%q�������JC{=^s�a�,V��R f���|lzB�&�����*�YO���r��w�6
�zx�NW,#�>�Fz��y3� ��8Ê��2�����#y�� %%��Q%�<K�J����~>��-����óV,Ƹ��U��}�]=�e���a���� �=̓����-�]�����@>�lt�ǥ�p焲��U�q��W��>�U0�=TȎ	X�{��&�Q���OԲ����0M\,7z_��R��D�q� lت�WB*����c�Vo���zB��.���n|>.����(��f��w���	��t�=�4!���!��Ӏ���p�lAA�A�����( u~K�o`��sy�o$[H�c�U��u�b�ɨ(@SXh
W�2ނ�����x��z<Wv"�(�\���FTn�4�v{6Q�z�Ƙ���+����!량��f�fj�1���������)ra�G� �:�4��m`ַL���9��c�8�������|����!Ra����&+�{�����E���-��ZZ1���@�c�LCv�D�$�θ�8�i4Ȇ|�^ߵ!-��n7�:���������3�&�Ȋ��o2�ܜ�/̔T4��.�^�4`٦O����s`��qd�,��x˛��|�!�)�̉?)��:M�R�� �(�#��R�Na�U{]ݺPg526T�ܭW�����>�D�f&�I\f��q��+�E�=��IG���V	���֬$[�#4�T���b5�/�y��%Q�[��D]fy�{��_�9����/��ŀB��֠/b(be�{i&����u��g�$3d�L�_:�p�n�� Q�_�o��(g24�ˌ�V�$Ͱ����i�*T�(L&95WE�2G����&Y�Qk�X���$K�a�I.��<�[�:��G��JU�#0��Lʜ����*�wl�5�Лjn0G~h�?1}θr��N�w�`|����"�{�jx�0���(=���������@\�����g��%�RV�־O�˓�]\�T�}F��rb&="�F?�xa�"���.���_��D�ǀ�ޱ���A�\��h��ת���9�b>^|�V��5@o��-0��,�6�S�a���X�m�:S��|K����Td�fF7Ҭ�-(�e�;j2Y:�� \���yZ����Z��3���Zx'	��U'�_A��MK����&q���v����$YJ�����|�^,A��`���e?9�>
�{�k���s�3O��g��&[�m9�]'n!5��-���Єa��Io䀉�J���b�4y����	�B}$Ox-�s2�<цưZǟ��(�Fyt%8�yc���
�fe�1�#�f�k�Z�]��c�fc�