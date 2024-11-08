import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import networkx as nx
from utils import find_value_according_index_list,natural_connectivity2,network_life
from GraphConstruct2 import location_graph
from sklearn.neighbors import NearestNeighbors

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

selected_node = [[] for _ in range(len(node_list))]
g = [[] for _ in range(len(node_list))]
A = [[] for _ in range(len(node_list))]
d = 1
natural_conn = [None] * (int(len(node_list)) + 1)
res_energy_avg = [None] * (int(len(node_list)) + 1)
communicate_circle = [None] * (int(len(node_list)) + 1)
x = [None] * (int(len(node_list)) + 1)
i = 0
# print(len(node_list))
for select_node in range(0,200,1):
    # print(select_node+1)
    x[i] = select_node+1
    selected_node[i] = node_list[:select_node+1]
    # print(selected_node[i] )
    fea_list = find_value_according_index_list(fea_o, selected_node[i])
    location_list = find_value_according_index_list(location, selected_node[i])
    # print(location_list)
# for select_node in range(len(node_list)):
#     selected_node[select_node] = node_list[:select_node+1]
#     fea_list = find_value_according_index_list(fea_o,selected_node[select_node])
#     location_list = find_value_according_index_list(location,selected_node[select_node])
    if x[i] > 1:
        g[i],A[i] = location_graph(location_list)
        natural_conn[i] = natural_connectivity2(g[i])
        communicate_circle[i],res_energy_avg[i] = network_life(g[i])
        # print(communicate_circle[i],res_energy_avg[i])
        # print(g[i])
    else:
        test = NearestNeighbors(radius=0.05)
        test.fit(location)  #
        A[i] = np.ones((1, 1))
        g[i] = nx.from_numpy_array(A[i])
        A[i] = nx.to_pandas_adjacency(g[i])
        # nx.draw(g[i], pos=location_list, with_labels=True, alpha=0.4, node_size=10, font_size=5)
    print(select_node)
    i = i+1
for select_node in range(201,int(len(node_list))+1,5):
    # print(select_node+1)
    x[i] = select_node+1
    selected_node[i] = node_list[:select_node+1]
    # print(selected_node[i] )
    fea_list = find_value_according_index_list(fea_o, selected_node[i])
    location_list = find_value_according_index_list(location, selected_node[i])
    # print(location_list)
# for select_node in range(len(node_list)):
#     selected_node[select_node] = node_list[:select_node+1]
#     fea_list = find_value_according_index_list(fea_o,selected_node[select_node])
#     location_list = find_value_according_index_list(location,selected_node[select_node])
    g[i],A[i] = location_graph(location_list)
    natural_conn[i] = natural_connectivity2(g[i])
    communicate_circle[i], res_energy_avg[i] = network_life(g[i])
    print(select_node)
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

# MSE
"""
多个节点特征怎么汇总
图特征怎么获得
"""
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

# # 示例数据
# y_true = [3, -0.5, 2, 7]
# y_pred = [2.5, 0.0, 2, 8]
#
# # 计算 MSE
# mse = mean_squared_error(y_true, y_pred)
# print(f"Mean Squared Error: {mse}")




fig, conn = plt.subplots(figsize=(8, 4))
AEC = conn.twinx()
## ax.axvline(max_natural_conn, c='#E89B9E', ls='--')
# ax.axhline(1.596723933094875, c='#E89B9E', ls='--')
line_CONN= conn.plot(x, natural_conn, marker = '.',markerfacecolor='white', label='Natural Connectivity', color='#d92523')
line_AEC = AEC.plot(x, res_energy_avg, markerfacecolor='white', label='res_energy_avg', color='#2e7ebb')

conn.set_ylabel('Natural Connectivity')
AEC.set_ylabel('res_energy_avg')
plt.title('Reference indicators')

# 设置坐标轴名
conn.set_xlabel('Number of nodes')

# 设置图例
lines = line_CONN + line_AEC
labels = [l.get_label() for l in lines]
conn.legend(lines, labels, loc='upper right')
# 统一两个 y 轴的比例
# conn.set_ylim(0, 1.7)
# AEC.set_ylim(0, 1.7)
plt.savefig('.\Reference indicators\_Reference indicator_test'+'.svg', format='svg',dpi=600)
plt.show()

