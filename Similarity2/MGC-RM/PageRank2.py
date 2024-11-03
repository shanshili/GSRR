import numpy as np
import pandas as pd
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
# 标号为数组下标
adjust =  1.463619555579498410e-04#75-329的数据 #1.949527258053421974e-05#159-328# #1.664528177166357636e-04#310-21#
with open('similarity score\gp_159_z_node_450_data_2000model_20241102_165518.txt','r') as f:
# with open('similarity score\gp_310_z_node_450_data_2000model_20241102_165317.txt','r') as f:
# with open('similarity score\zGP_75_node_450_data_2000model_20241102_211834.txt','r') as f:
	for line in f:
		weights.append(float(list(line.strip('\n').split(','))[0]))
node = 450
data_num = 2000
adj_ori = np.load('./origin_data/adj_ori'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', allow_pickle=True)
location_file = np.load('./origin_data/location'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', allow_pickle=True)
A = adj_ori['arr_0']
G = nx.Graph(adj_ori['arr_0'])
# print('location',location_file.files,location_file['arr_0'].shape)
location = location_file['arr_0']
lon = location[0]
lat = location[1]



def getGm(A):
    '''
    功能：求状态转移概率矩阵Gm
    @A：网页链接图的邻接矩阵
    '''
    Gm = []
    for i in range(len(A)):
        cnt = 0
        for j in range(len(A[i])):
            if A[i][j] != 0:
                cnt += 1
        tran_prob = 1 / cnt  # 转移概率
        Gm_tmp = []
        for j in range(len(A[i])):
            Gm_tmp.append(tran_prob * A[i][j])
        Gm.append(Gm_tmp)
    Gm = np.transpose(Gm)
    return Gm


def WeightedPageRank(M, weights, N, T=300, eps=1e-6, beta=0.85):
    R = np.ones(N) / N
    teleport = np.ones(N) / N
    for time in range(T):
        R_new = beta * np.dot(M, weights * R) + (1 - beta) * teleport
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


weights = weights / np.sum(weights)
# print(weights)
print(np.sum(weights))

# G_m = getGm(A)

D = np.diag(np.power(np.sum(A, axis=1), -1))
M = np.dot(A, D)
# print(np.sum(M, axis=0))
# print(G_m.shape)
contribution_weights = WeightedPageRank(M, weights, N=node, T=300)
# contribution_weights = PageRank(M, N=144, T=300)

node_list = np.argsort(-contribution_weights)
print(node_list.tolist())

contribution_weights = contribution_weights / np.sum(contribution_weights)
# contribution_weights = np.exp(contribution_weights) / np.sum(np.exp(contribution_weights))

print(contribution_weights.tolist())

# print(np.sum(contribution_weights))

plt.plot(range(1, node+1), contribution_weights, '-o')
plt.xlabel('Sensor node')
plt.ylabel('Contribution weights')
plt.show()


plt.scatter(lon, lat, c=contribution_weights, cmap='plasma_r')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar()
plt.title('Contribution weights')
plt.show()



# selected_node = similarity.argsort()[:15]
selected_node = node_list[:15]

plt.scatter(lon, lat, c='gainsboro')
plt.scatter(lon[selected_node], lat[selected_node], c='#7e2f8e')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('MGC-RM')
# plt.savefig(fname="../figure/optimal_distribution_144.svg", format="svg")
plt.show()

