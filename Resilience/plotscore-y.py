import sys
import os

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 找到项目的根目录（假设 utils 和 MGC-RM 是同级目录）
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# 将项目根目录添加到 Python 的搜索路径中
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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


#'softsort_normal_epoch_350_lr_1e-06_20241127_180601'
# 'softsort_normal_epoch_250_lr_1e-05_20241127_184058'
# 'softsort_normal_epoch_100_lr_3e-07_20241128_101450'
# 'softsort_normal_epoch_410_lr_1e-07_20241203_140404'
# softsort_normal_epoch_210_lr_1e-07_20241203_114951
# softsort_normal_epoch_400_lr_1e-07_20241202_101508
#softsort_normal_epoch_210_lr_1e-07_20241203_122747
file_softsort_normal = 'softsort_normal_epoch_210_lr_1e-07_20241203_115724'


softsort_normal = []
with open('./scores_save/'+file_softsort_normal+'.txt', 'r') as f:
    for line in f:
        softsort_normal.append(float(list(line.strip('\n').split(','))[0]))

# criticality_scores_normal 越小越重要
softsort_normal_index = np.argsort(softsort_normal)   # 从小到大的索引，将数组从小到大排序后，每个元素在原数组中的位置。
r_select = 114
# y
select_index = softsort_normal_index[:r_select]


# def stretch_first_20(data, power=50):
#     """
#     对数据的最后20个值应用非线性变换（如幂函数）。
#
#     参数:
#         data: list or ndarray, 输入的数据 (长度为114)
#         power: float, 幂次，默认为2
#
#     返回:
#         ndarray, 变换后的数据
#     """
#     num = 5
#
#     data_array = np.array(data)
#     transformed_data = data_array.copy()
#
#     if len(transformed_data) >= num:
#         # 应用幂函数变换到最后20个值上
#         transformed_data[-num:] = transformed_data[-num:] ** power
#
#         # 如果需要，确保转换后的数据仍在合理范围内（假设原始数据范围是 [0, 1]）
#         transformed_data[-num:] = np.clip(transformed_data[-num:], 0, 1)
#
#     return transformed_data
#
# softsort_normal2 =  stretch_first_20(softsort_normal)

 # selected_node #cfb7d5  marker='D'
colors = 'Greens_r'
plt.scatter(un_location_list_a[select_index,0],
            un_location_list_a[select_index,1],
            s=15,c=[softsort_normal[i] for i in select_index],cmap= colors)
plt.colorbar()
# for i in range(len(select_index)):
#     plt.text(un_location_list_a[select_index[i],0],
#             un_location_list_a[select_index[i],1], i+1, ha='right',fontsize=5)  # 使用plt.text添加文本
plt.scatter(location_list_a[:,0], location_list_a[:,1] ,s=20, c='#f5b7b1',label='Optimized nodes')
plt.scatter(un_location_list_a[select_index[0],0],
           un_location_list_a[select_index[0],1], s=30, c='#e74c3c',label='Optimal backup node')  # selected_node #d400fe ,marker='*'
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Nodes resilience rank forecast')
plt.legend(loc='lower left')
plt.savefig('./scores_save/plotscore-y/y_select_'+str(r_select)+'_'+file_softsort_normal+'4.svg', format='svg')
plt.show()


plt.scatter(lon, lat,s = 15, c='gainsboro', label='All nodes')
plt.scatter(location_list_a[:,0], location_list_a[:,1],s=30, c='#7e2f8e',label='Optimized nodes')  # selected_node
plt.scatter(un_location_list_a[select_index[0],0],
            un_location_list_a[select_index[0],1], s=30, c='#d400fe',label='Optimal backup node')  # selected_node
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Final optimal nodes')
plt.legend(loc='upper right')
# plt.savefig('./scores_save/plotscore-y/all'+str(r_select)+'_'+file_softsort_normal+'.svg', format='svg')
plt.show()




