from utils.utils import find_value_according_index_list
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
wpr_rank = []
with open('../MGC-RM/WPR_result/_node_list_GP_75_node_450_data_2000.txt', 'r') as f:
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
# file_scores = 'scores_epoch_350_lr_1e-06_20241127_180601'# 'scores_epoch_200_lr_1e-06_20241125_105756'
#'softsort_normal_epoch_350_lr_1e-06_20241127_180601'
# 'softsort_normal_epoch_250_lr_1e-05_20241127_184058'
# 'softsort_normal_epoch_100_lr_3e-07_20241128_101450'
# 'softsort_normal_epoch_410_lr_1e-07_20241203_140404'
file_softsort_normal = 'softsort_normal_epoch_100_lr_3e-07_20241128_101450'
criticality_scores_normal = []
# R_Rg_tensor = []
with open('./robustness_score/'+file_criticality_scores_normal+'.txt', 'r') as f:
    for line in f:
        criticality_scores_normal.append(float(list(line.strip('\n').split(','))[0]))
# with open('./robustness_score/'+file_R_Rg_tensor+'.txt', 'r') as f:
#     for line in f:
#         R_Rg_tensor.append(float(list(line.strip('\n').split(','))[0]))


# criticality_scores_normal 越大越重要
criticality_scores_normal_index = np.argsort(criticality_scores_normal)
r_select = 99
# r
select_index = criticality_scores_normal_index[:r_select]


colors = 'autumn'
plt.scatter(un_location_list_a[select_index,0],
            un_location_list_a[select_index,1],
            s=30,c=[criticality_scores_normal[i] for i in select_index],cmap= colors)
plt.colorbar()
for i in range(len(select_index)):
    plt.text(un_location_list_a[select_index[i],0],
            un_location_list_a[select_index[i],1], i+1, ha='right',fontsize=5)  # 使用plt.text添加文本
plt.scatter(un_location_list_a[select_index[0],0],
            un_location_list_a[select_index[0],1], s=35, c='#4fc3f7')  # selected_node
plt.scatter(location_list_a[:,0], location_list_a[:,1], s=30, c='#abebc6')  # selected_node
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Rg')
plt.savefig('./scores_save/plotscore-r/r_select_'+str(r_select)+'.svg', format='svg')
plt.show()
