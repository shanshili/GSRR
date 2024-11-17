import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import networkx as nx
from utils import (find_value_according_index_list,
                   natural_connectivity2,network_life,
                   MSE_node_feature,mean_squared_error,DS2,DS3,
                   MSE_all_node_feature,calculate_aec)
from GraphConstruct2 import location_graph
from sklearn.neighbors import NearestNeighbors
from model import AutoEncoder

# 全局修改字体
config = {
            "font.family": 'Times New Roman',
            "font.size": 10.5,
            # "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            # "font.serif": ['SimSun'],#宋体
            'axes.unicode_minus': False # 处理负号，即-号
         }
rcParams.update(config)

"""
根据重要性依次选择选择节点加入，构成1-450点的450个图，观察各性能曲线趋势
很多鲁棒性的指标就是反着来，依次删掉节点，也是要对比450个图
先写衡量指标，依次计算450个数值的代码，然后画在一张图里

设定一个指标步距，比如450/30,15个点看趋势有没有明显的区分
"""
node = 450
data_num = 2000
adj_ori = np.load('./origin_data/adj_ori'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', allow_pickle=True)
location_file = np.load('./origin_data/location'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', allow_pickle=True)
fea_ori = np.load('./origin_data/fea_ori'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', allow_pickle=True)
A_o = adj_ori['arr_0']
G_o = nx.Graph(adj_ori['arr_0'])
location = location_file['arr_0']
fea_o = fea_ori['arr_0']
lon = location[:,0]
lat = location[:,1]

# 读取节点排序
node_list=[]
with open('WPR_result\_node_list_GP_75_node_450_data_2000.txt','r') as f:
    for line in f:
        node_list.append(int(list(line.strip('\n').split(','))[0]))

# G_0 中最重要节点的拓扑特征编码
encode_o = MSE_node_feature(G_o,node_list[0])
# padded_vectors = MSE_all_node_feature(G_o,node_list)
# print(padded_vectors)
# print(location[node_list[0]])

selected_node = [[] for _ in range(len(node_list))]
g = [[] for _ in range(len(node_list))]
A = [[] for _ in range(len(node_list))]
natural_conn = [None] * (int(len(node_list)) + 1)
res_energy_avg = [None] * (int(len(node_list)) + 1)
communicate_circle = [None] * (int(len(node_list)) + 1)
AEC_2 = [None] * (int(len(node_list)) + 1)
mse = [None] * (int(len(node_list)) + 1)
mse_all_node = [None] * (int(len(node_list)) + 1)
Ds = [None] * (int(len(node_list)) + 1)
x = [None] * (int(len(node_list)) + 1)
i = 0 # 采样计数
"""
小图密集采样显示
"""
for select_node in range(0,100,1):
    x[i] = select_node+1 # 已选择节点数目
    selected_node[i] = node_list[:select_node+1]
    fea_list = find_value_according_index_list(fea_o, selected_node[i])
    location_list = find_value_according_index_list(location, selected_node[i])
    if x[i] > 1:
        g[i],A[i] = location_graph(location_list)

        natural_conn[i] = natural_connectivity2(g[i])
        # communicate_circle[i],res_energy_avg[i] = network_life(g[i])
        AEC_2[i] = calculate_aec(g[i],location_list)
        # encode = MSE_node_feature(g[i],0)
        # mse[i] = mean_squared_error(encode_o,encode)
        # mse_all_node[i] = MSE_all_node_feature(g[i],selected_node[i])
        # mse[i] = mean_squared_error(padded_vectors, mse_all_node[i])
        # Ds[i] = DS(np.array(location_list)[:, 0],np.array(location_list)[:, 1], select_node + 1)
        Ds[i] = DS2(g[i], select_node + 1)
        # Ds[i] = DS3(location_list, select_node + 1)
        # print(mse[i])
        # print(mse_all_node[i])
        # print(communicate_circle[i],res_energy_avg[i])
    else:  # 仅选择一个节点时
        test = NearestNeighbors(radius=0.05)
        test.fit(location)  #
        A[i] = np.ones((1, 1))
        g[i] = nx.from_numpy_array(A[i])
        A[i] = nx.to_pandas_adjacency(g[i])

        natural_conn[i] = natural_connectivity2(g[i])
        # communicate_circle[i], res_energy_avg[i] = network_life(g[i])
        AEC_2[i] = calculate_aec(g[i], location_list)
        # encode = MSE_node_feature(g[i],0)
        # mse[i] = mean_squared_error(encode_o,encode)
        # mse_all_node[i] = MSE_all_node_feature(g[i], selected_node[i])
        # mse[i] = mean_squared_error(padded_vectors, mse_all_node[i])
        # print(np.array(location_list)[:, 0])
        # Ds[i] = DS(np.array(location_list)[:, 0],np.array(location_list)[:, 1], select_node + 1)
        Ds[i] = DS2(g[i], select_node + 1)
        # Ds[i] = DS3(location_list, select_node + 1)

        # print(mse[i])
        # print(mse_all_node[i])
        # nx.draw(g[i], pos=location_list, with_labels=True, alpha=0.4, node_size=10, font_size=5)
    print(x[i])
    i = i+1  # 采样计数
"""
大图粗略采样显示
"""
for select_node in range(101,int(len(node_list))+1,5):
    x[i] = select_node+1
    selected_node[i] = node_list[:select_node+1]
    fea_list = find_value_according_index_list(fea_o, selected_node[i])
    location_list = find_value_according_index_list(location, selected_node[i])
    g[i],A[i] = location_graph(location_list)

    # natural_conn[i] = natural_connectivity2(g[i])
    # communicate_circle[i], res_energy_avg[i] = network_life(g[i])
    # AEC_2[i] = calculate_aec(g[i], location_list)
    # encode = MSE_node_feature(g[i],0)
    # mse[i] = mean_squared_error(encode_o,encode)
    # mse_all_node[i] = MSE_all_node_feature(g[i], selected_node[i])
    # mse[i] = mean_squared_error(padded_vectors, mse_all_node[i])
    # Ds[i] = DS(np.array(location_list)[:, 0],np.array(location_list)[:, 1], select_node + 1)
    # Ds[i] = DS2(g[i], select_node + 1)
    # Ds[i] = DS3(location_list, select_node + 1)
    # print(mse[i])
    # print(mse_all_node[i])
    # print(x[i])
    i = i+1

## print(natural_conn)
# print(communicate_circle,res_energy_avg)
# natural_conn_cleaned = [value for value in natural_conn if value is not None]
# print(natural_conn_cleaned)
# max_natural_conn = natural_conn.index(max(natural_conn_cleaned))
# print(max(natural_conn_cleaned))
##  max_natural_conn = natural_conn.index(1.596723933094875)+1
# print(max_natural_conn)
# print(x)
# print(mse)

"""
多个节点特征怎么汇总
图特征怎么获得
"""





fig, conn = plt.subplots(figsize=(8, 4))
# AEC = conn.twinx()
# cc = conn.twinx()
mse_ax = conn.twinx()
aec_2 = conn.twinx()
ds_ax = conn.twinx()
# mse_all_ax = conn.twinx()

# cc.spines['right'].set_position(('outward', 40))
# mse_ax.spines['right'].set_position(('outward', 40))
aec_2.spines['right'].set_position(('outward', 30))
ds_ax.spines['right'].set_position(('outward', 75))
# mse_all_ax.spines['right'].set_position(('outward', 80))


line_CONN= conn.plot(x, natural_conn, marker = '.',markerfacecolor='white', label='NC', color='#d92523')
# line_AEC = AEC.plot(x, res_energy_avg, markerfacecolor='white', label='res_energy_avg', color='#2e7ebb')
# line_cc = cc.plot(x, communicate_circle,marker = '.', markerfacecolor='white', label='communicate_circle', color='#00FF00')
line_AEC2 = aec_2.plot(x, AEC_2, marker = '.',markerfacecolor='white', label='AEC', color='#2e7ebb')
# line_mse = mse_ax.plot(x, mse,marker = '.',markerfacecolor='white', label='MSE', color='#FFA500')
line_ds = ds_ax.plot(x, Ds,marker = '.',markerfacecolor='white', label='Ds', color='#00FFFF')
# line_mse_all = mse_all_ax.plot(x, mse_all_node,marker = '.', markerfacecolor='white', label='mse_all', color='#FF1493')

# conn.axvline(6, c='#E89B9E', ls='--')
# plt.text(6,0,'6+')
# conn.axvline(30, c='#6495ED', ls='--')
# plt.text(30,0,'30-')
# conn.axvline(12, c='#E89B9E', ls='--')
# plt.text(12,0,'12+')
# conn.axvline(24, c='#00FF00', ls='--')
# plt.text(24,0,'24')
conn.axvline(14, c='#00FF00', ls='--')
plt.text(14,0,'14')

conn.set_ylabel('NC')
# AEC.set_ylabel('res_energy_avg')
# cc.set_ylabel('communicate_circle')
aec_2.set_ylabel('AEC')
# mse_ax.set_ylabel('MSE')
ds_ax.set_ylabel('Ds')
# mse_all_ax.set_ylabel(' mse2')


# 设置坐标轴名
conn.set_xlabel('Number of nodes')
plt.title('Reference indicators')


# 设置图例
lines =  line_CONN+line_AEC2+line_ds
labels = [l.get_label() for l in lines]
conn.legend(lines, labels, loc='right')
# 统一两个 y 轴的比例
# conn.set_ylim(0, 1.7)
# AEC.set_ylim(0, 1.7)
plt.savefig('.\Reference indicators\select-indicators5'+'.svg', format='svg',dpi=600)
plt.show()

