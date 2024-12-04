import sys

# 将外部包的路径添加到 sys.path
sys.path.append('D:\Tjnu-p\ML-learning\similarity2\MGC-RM')
# 现在可以导入外部包了
from utils import find_value_according_index_list, robustness_score
import numpy as np
import networkx as nx
from matplotlib import rcParams
import matplotlib.pyplot as plt

# 全局修改字体
config = {
    "font.family": 'Times New Roman',
    "font.size": 10.5,
    # "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    # "font.serif": ['SimSun'],#宋体
    'axes.unicode_minus': False  # 处理负号，即-号
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
"""
在选择图的基础上获得鲁棒图
节点临界分数：鲁棒图中去掉该节点，图鲁棒性大幅下降
"""
selected_node = wpr_rank[:select_node]
fea_list = find_value_according_index_list(fea_o, selected_node)
location_list = find_value_according_index_list(location, selected_node)
unselected_node = wpr_rank[select_node + 1:114]
un_fea_list = find_value_according_index_list(fea_o, unselected_node)
un_location_list = find_value_according_index_list(location, unselected_node)
location_list_a = np.array(location_list)
un_location_list_a = np.array(un_location_list)

file_criticality_scores_normal = 'criticality_scores_normal_99'
file_R_Rg_tensor = 'R_Rg_tensor_99'
file_scores = 'scores_epoch_410_lr_1e-07_20241203_140404'# 'scores_epoch_200_lr_1e-06_20241125_105756'
file_softsort_normal = 'softsort_normal_epoch_410_lr_1e-07_20241203_140404'# 'softsort_normal_epoch_200_lr_1e-06_20241125_105756'
criticality_scores_normal = []
R_Rg_tensor = []
with open('./robustness_score/'+file_criticality_scores_normal+'.txt', 'r') as f:
    for line in f:
        criticality_scores_normal.append(float(list(line.strip('\n').split(','))[0]))
with open('./robustness_score/'+file_R_Rg_tensor+'.txt', 'r') as f:
    for line in f:
        R_Rg_tensor.append(float(list(line.strip('\n').split(','))[0]))

scores = []
softsort_normal = []
with open('./scores_save/'+file_scores+'.txt', 'r') as f:
    for line in f:
        scores.append(float(list(line.strip('\n').split(','))[0]))
with open('./scores_save/'+file_softsort_normal+'.txt', 'r') as f:
    for line in f:
        softsort_normal.append(float(list(line.strip('\n').split(','))[0]))


# softsort_normal = np.argsort(softsort_normal)
# criticality_scores_normal = np.argsort(criticality_scores_normal)



colors = 'Greens_r'
plt.scatter(location_list_a[:,0], location_list_a[:,1], s=20, c='#f44336')  # selected_node
plt.scatter(un_location_list_a[:,0], un_location_list_a[:,1], s=15, c=scores, cmap= colors)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar()
plt.title('scores')
plt.savefig('./scores_save/3'+file_scores+'.svg', format='svg')
plt.show()

plt.scatter(location_list_a[:,0], location_list_a[:,1], s=20, c='#f44336')  # selected_node
plt.scatter(un_location_list_a[:,0], un_location_list_a[:,1], s=15, c=softsort_normal, cmap=colors)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar()
plt.title('softsort_normal')
plt.savefig('./scores_save/3'+file_softsort_normal+'.svg', format='svg')
plt.show()

plt.scatter(location_list_a[:,0], location_list_a[:,1], s=20, c='#f44336')  # selected_node
plt.scatter(un_location_list_a[:,0], un_location_list_a[:,1], s=15, c=R_Rg_tensor, cmap=colors)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar()
plt.title('R_Rg_tensor')
plt.savefig('./robustness_score/3'+file_R_Rg_tensor+'.svg', format='svg')
plt.show()

plt.scatter(location_list_a[:,0], location_list_a[:,1], s=20, c='#f44336')  # selected_node
plt.scatter(un_location_list_a[:,0], un_location_list_a[:,1], s=15, c=criticality_scores_normal, cmap=colors)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar()
plt.title('criticality_scores_normal')
plt.savefig('./robustness_score/3'+file_criticality_scores_normal+'.svg', format='svg')
plt.show()
