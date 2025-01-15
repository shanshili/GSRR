import sys
import os

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 找到项目的根目录（假设 utils 和 MGC-RM 是同级目录）
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# 将项目根目录添加到 Python 的搜索路径中
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.GraphConstruct import data_color_graph3
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 全局修改字体
config = {
            "font.family": 'Times New Roman',
            "font.size": 10.5,
            # "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            # "font.serif": ['SimSun'],#宋体
            'axes.unicode_minus': False # 处理负号，即-号
         }
rcParams.update(config)

# 读取图结构
weights=[]
with open('similarity score/zGP_75_node_450_data_2000model_20241102_211834.txt', 'r') as f:
	for line in f:
		weights.append(float(list(line.strip('\n').split(','))[0]))
Gpn = 75
node = 450
data_num = 2000
select_node = 14
adj_ori = np.load('./origin_data/adj_ori'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', allow_pickle=True)
location_file = np.load('./origin_data/location'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', allow_pickle=True)
A = adj_ori['arr_0']
G_o = nx.Graph(A)
G = nx.Graph(adj_ori['arr_0'])
location = location_file['arr_0']
lon = location[:,0]
lat = location[:,1]


def WeightedPageRank(M, weights, N, T=300, eps=1e-6, beta=0.85):
    """
    R : rank 每次迭代
    N : 节点个数
    M : 概率转移矩阵
    beta : 阻尼因子，控制随机跳转的概率
    T : 迭代次数
    eps ： 迭代收敛限度
    """
    R = np.ones(N) / N  # 向量
    teleport = np.ones(N) / N
    for time in range(T):
        R_new = beta * np.dot(M, weights * R) + (1 - beta) * teleport   #weights * R 扩展矩阵 且1/N
        # R_new = np.dot(M, weights * R)

        if np.linalg.norm(R_new - R) < eps:
            break
        R = R_new.copy()
    return R_new



def PageRank(M, N, T=300, eps=1e-6, beta=0.85):
    R = np.ones(N) / N
    teleport = np.ones(N) / N
    for time in range(T):
        R_new = beta * np.dot(M, R) + (1 - beta) * teleport
        # R_new = np.dot(M, weights * R)
        if np.linalg.norm(R_new - R) < eps:
            break
        R = R_new.copy()
    return R_new


weights = weights / np.sum(weights)  # 归一化

D = np.diag(np.power(np.sum(A, axis=1), -1))
M = np.dot(A, D)   # 概率转移矩阵
contribution_weights = WeightedPageRank(M, weights, N=node, T=300)
node_list = np.argsort(-contribution_weights)
contribution_weights = contribution_weights / np.sum(contribution_weights) # 归一化
# print(node_list,contribution_weights)

# 重要性从大到小排序
np.savetxt('./WPR_result/_node_list_'+ 'GP_'+str(Gpn)+'_node_'+str(node)+'_data_'+str(data_num)+'.txt', node_list,fmt='%d')


# print(np.sum(contribution_weights))
plt.figure()
plt.plot(range(1, node+1), contribution_weights, '-o')
plt.xlabel('Sensor node')
plt.ylabel('Contribution weights')
## plt.savefig('WPR_result\Gp' + str(Gpn) +'_Contribution_weights_1' +'.svg', format='svg')
plt.show()

plt.scatter(lon, lat, s = 10,c=contribution_weights, cmap='plasma_r')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar()
plt.title('Contribution weights')
## plt.savefig('WPR_result\Gp' + str(Gpn) +'_Contribution_weights_2' +'.svg', format='svg')
plt.show()


selected_node = node_list[:select_node]    # 倒数最重要的select_node个
plt.scatter(lon, lat,s = 15, c='gainsboro', label='All nodes')
plt.scatter(lon[selected_node], lat[selected_node], s = 30,c='#7e2f8e',label='Optimized nodes')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Optimized nodes subset')
plt.legend()
plt.savefig('WPR_result\Gp' + str(Gpn) +'_MGC-RM'+'select'+str(select_node) +'2.svg', format='svg')
plt.show()

plt.scatter(lon, lat,s = 15, c='gainsboro', label='All nodes')
plt.scatter(lon[selected_node], lat[selected_node], s = 30,c='#7e2f8e',label='Optimized nodes')
plt.scatter(lon[node_list[select_node] ], lat[node_list[select_node] ], s = 30,c='#d400fe',label='Contribution weights backup node')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('')
plt.legend()
plt.savefig('WPR_result\Gp' + str(Gpn) +'_MGC-RM'+'select'+str(select_node) +'3.svg', format='svg')
plt.show()

data_color_graph3(contribution_weights,G_o,location,'plasma_r',False)


