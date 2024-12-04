from matplotlib import rcParams
import numpy as np
import networkx as nx
import sys
sys.path.append('F:\Tjnu-p\ML-learning\similarity2\MGC-RM')
from utils import (find_value_according_index_list,
                   natural_connectivity2,network_life,
                   MSE_node_feature,mean_squared_error,DS2,DS3,
                   MSE_all_node_feature,calculate_aec,
                   find_value_according_index_list, robustness_score)
from GraphConstruct2 import location_graph
import copy

# 全局修改字体
config = {
            "font.family": 'Times New Roman',
            "font.size": 10.5,
            # "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            # "font.serif": ['SimSun'],#宋体
            'axes.unicode_minus': False # 处理负号，即-号
         }
rcParams.update(config)



# 读取数据
weights = []
wpr_rank = []
with open('..\MGC-RM\similarity score\zGP_75_node_450_data_2000model_20241102_211834.txt', 'r') as f:
    for line in f:
        weights.append(float(list(line.strip('\n').split(','))[0]))
with open('..\MGC-RM\WPR_result\_node_list_GP_75_node_450_data_2000.txt', 'r') as f:
    for line in f:
        wpr_rank.append(int(list(line.strip('\n').split(','))[0]))
Gpn = 75
node = 450
data_num = 2000
select_node = 14

adj_ori = np.load('../MGC-RM/origin_data/adj_ori' + '_node_' + str(node) + '_data_' + str(data_num) + '.npz',
                  allow_pickle=True)
location_file = np.load('../MGC-RM/origin_data/location' + '_node_' + str(node) + '_data_' + str(data_num) + '.npz',
                        allow_pickle=True)
fea_ori = np.load('../MGC-RM/origin_data/fea_ori' + '_node_' + str(node) + '_data_' + str(data_num) + '.npz',
                  allow_pickle=True)
fea_o = fea_ori['arr_0']
A = adj_ori['arr_0']
G = nx.Graph(adj_ori['arr_0'])
location = location_file['arr_0']
lon = location[:, 0]
lat = location[:, 1]


file_criticality_scores_normal = 'criticality_scores_normal_99'
file_R_Rg_tensor = 'R_Rg_tensor_99'
file_scores = 'scores_epoch_210_lr_1e-07_20241203_122747'# 'scores_epoch_200_lr_1e-06_20241125_105756'
#'softsort_normal_epoch_350_lr_1e-06_20241127_180601'
# 'softsort_normal_epoch_250_lr_1e-05_20241127_184058'
# 'softsort_normal_epoch_100_lr_3e-07_20241128_101450'
# 'softsort_normal_epoch_410_lr_1e-07_20241203_140404'
#
file_softsort_normal = 'softsort_normal_epoch_210_lr_1e-07_20241203_122604'
criticality_scores_normal = []
R_Rg_tensor = []
with open('./robustness_score/'+file_criticality_scores_normal+'.txt', 'r') as f:
    for line in f:
        criticality_scores_normal.append(float(list(line.strip('\n').split(','))[0]))
with open('./robustness_score/'+file_R_Rg_tensor+'.txt', 'r') as f:
    for line in f:
        R_Rg_tensor.append(float(list(line.strip('\n').split(','))[0]))

# print(criticality_scores_normal)
# print(R_Rg_tensor)
scores = []
softsort_normal = []
with open('./scores_save/'+file_scores+'.txt', 'r') as f:
    for line in f:
        scores.append(float(list(line.strip('\n').split(','))[0]))
with open('./scores_save/'+file_softsort_normal+'.txt', 'r') as f:
    for line in f:
        softsort_normal.append(float(list(line.strip('\n').split(','))[0]))

# print(scores)
# print(softsort_normal)
# criticality_scores_normal 越小越重要
softsort_normal_index = np.argsort(softsort_normal)   # 从小到大的索引，将数组从小到大排序后，每个元素在原数组中的位置。
r_select = 1
select_index = softsort_normal_index[:1]
print(select_index)
for i in select_index:
    print(i)
    print(softsort_normal[i])
optimize_node_index = wpr_rank[:select_node]
print(optimize_node_index)
optimize_fea_list = find_value_according_index_list(fea_o, optimize_node_index)
optimize_location_list = find_value_according_index_list(location, optimize_node_index)
unselected_node = wpr_rank[select_node + 1:114]
un_fea_list = find_value_according_index_list(fea_o, unselected_node)
un_location_list = find_value_according_index_list(location, unselected_node)

R_node_index = softsort_normal_index[:r_select]
r_fea_list = find_value_according_index_list(un_fea_list, R_node_index)
r_location_list = find_value_according_index_list(un_location_list, R_node_index)

location = copy.deepcopy(optimize_location_list)
location.append(r_location_list[0])
fea = copy.deepcopy(optimize_fea_list)
fea.append(r_fea_list[0])
# print(len(optimize_fea_list))
# print(len(fea))

location_a = np.array(location)
fea_a = np.array(fea)
optimize_location_list_a = np.array(optimize_location_list)
optimize_fea_list_a = np.array(optimize_fea_list)


g_o, A_o = location_graph(optimize_location_list)
g_r,A_r = location_graph(location)
Rg_o = robustness_score(g_o)
Rg_r = robustness_score(g_r)
print('Rg_o',Rg_o)
print('Rg_r',Rg_r)
natural_conn_o = natural_connectivity2(g_o)
natural_conn_r = natural_connectivity2(g_r)
print('nc_o',natural_conn_o)
print('nc_r',natural_conn_r)
AEC_o = calculate_aec(g_o,optimize_location_list)
AEC_r = calculate_aec(g_r,location)
print('AEC_o',AEC_o)
print('AEC_r',AEC_r)
Ds_o = DS2(g_o, len(optimize_location_list))
Ds_r = DS2(g_r, len(location))
print('Ds_o',Ds_o)
print('Ds_r',Ds_r)