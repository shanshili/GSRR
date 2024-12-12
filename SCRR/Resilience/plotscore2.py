from SCRR.utils.utils import find_value_according_index_list, robustness_score
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
# 'scores_epoch_100_lr_3e-07_20241128_100535'
# 'scores_epoch_200_lr_1e-07_20241130_212934'
# 'scores_epoch_250_lr_1e-05_20241127_184058'
file_scores = 'scores_epoch_200_lr_1e-06_20241130_224159'
# 'softsort_normal_epoch_100_lr_3e-07_20241128_101450'
# 'softsort_normal_epoch_200_lr_1e-07_20241130_212934'
# 'softsort_normal_epoch_250_lr_1e-05_20241127_184058'
file_softsort_normal = 'softsort_normal_epoch_200_lr_1e-06_20241130_224159'
criticality_scores_normal = []
R_Rg_tensor = []
with open('./robustness_score/' + file_criticality_scores_normal + '.txt', 'r') as f:
    for line in f:
        criticality_scores_normal.append(float(list(line.strip('\n').split(','))[0]))
with open('./robustness_score/' + file_R_Rg_tensor + '.txt', 'r') as f:
    for line in f:
        R_Rg_tensor.append(float(list(line.strip('\n').split(','))[0]))

scores = []
softsort_normal = []
with open('./scores_save/' + file_scores + '.txt', 'r') as f:
    for line in f:
        scores.append(float(list(line.strip('\n').split(','))[0]))
with open('./scores_save/' + file_softsort_normal + '.txt', 'r') as f:
    for line in f:
        softsort_normal.append(float(list(line.strip('\n').split(','))[0]))

R_Rg_tensor = sorted(R_Rg_tensor, reverse=False)
criticality_scores_normal = sorted(criticality_scores_normal, reverse=False)
scores = sorted(scores, reverse=False)
# print(scores)
softsort_normal = sorted(softsort_normal, reverse=False)

fig, r_rg = plt.subplots()
rank = r_rg.twinx()
predict = r_rg.twinx()
predict_rank = r_rg.twinx()
line_rg = r_rg.plot(range(len(R_Rg_tensor)), R_Rg_tensor, marker='.', markerfacecolor='white', label='Rg',
                    color='#3498db')
line_rank = rank.plot(range(len(criticality_scores_normal)), criticality_scores_normal, marker='.',
                      markerfacecolor='white', label='Rg rank', color='#1abc9c')
line_predict = predict.plot(range(len(scores)), scores, marker='.', markerfacecolor='white', label='Predict scores',
                            color='#e74c3c')
line_predict_rank = predict_rank.plot(range(len(softsort_normal)), softsort_normal, marker='.', markerfacecolor='white',
                                      label='Predict rank', color='#e67e22')
# r_rg.set_ylim(-0.1, 1)  # 固定左侧 y 轴的范围
# predict.set_ylim(-0.1, 1)  # 固定左侧 y 轴的范围
# rank.set_ylim(-0.1, 1)  # 固定左侧 y 轴的范围
# predict_rank.set_ylim(-0.1, 1)  # 固定左侧 y 轴的范围
# # 如果想要保持一定的比例，可以通过设置相同的刻度间隔实现
# r_rg.yaxis.set_ticks(np.arange(-2.5, 2.6, 1))
# ax2.yaxis.set_ticks(np.arange(-3.5, 3.6, 1))
predict.set_ylabel('Predict scores')
predict_rank.set_ylabel('Predict rank')
r_rg.set_ylabel('Rg')
rank.set_ylabel('Rg rank')
r_rg.set_xlabel('Number of nodes')
plt.title('Resilience scores')
lines = line_rg + line_rank + line_predict + line_predict_rank
labels = [l.get_label() for l in lines]
r_rg.legend(lines, labels, loc='lower right')
plt.savefig('./scores_save/plotscore2-linechart/' + file_scores + '.svg', format='svg')
plt.show()
