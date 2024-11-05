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
with open('similarity score\zGP_75_node_450_data_2000model_20241102_211834.txt','r') as f:
	for line in f:
		weights.append(float(list(line.strip('\n').split(','))[0]))
Gpn = 75
node = 450
data_num = 2000
select_node = 45
adj_ori = np.load('./origin_data/adj_ori'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', allow_pickle=True)
location_file = np.load('./origin_data/location'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', allow_pickle=True)
A = adj_ori['arr_0']
G = nx.Graph(adj_ori['arr_0'])
# print('location',location_file.files,location_file['arr_0'].shape)
location = location_file['arr_0']
lon = location[:,0]
lat = location[:,1]



def getGm(A):
    '''
    功能：求状态转移概率矩阵Gm
    @A：网页链接图的邻接矩阵 #扰动图的邻接矩阵
    '''
    Gm = []
    print('shape ',A.shape())
    print('len(A) ',len(A))
    print('range(len(A)) ',range(len(A)))
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
# print(weights)
# print('Sum_weights: ',np.sum(weights))

# G_m = getGm(A)

D = np.diag(np.power(np.sum(A, axis=1), -1))
M = np.dot(A, D)   # 概率转移矩阵
# print(np.sum(M, axis=0))
# print(G_m.shape)
contribution_weights = WeightedPageRank(M, weights, N=node, T=300)
# contribution_weights = PageRank(M, N=144, T=300)

node_list = np.argsort(-contribution_weights)
# print('Node_list:\n',node_list.tolist())

contribution_weights = contribution_weights / np.sum(contribution_weights) # 归一化
# contribution_weights = np.exp(contribution_weights) / np.sum(np.exp(contribution_weights))
# print('L1_contribution_weights:\n',contribution_weights.tolist())

# print(np.sum(contribution_weights))
plt.figure()
plt.plot(range(1, node+1), contribution_weights, '-o')
plt.xlabel('Sensor node')
plt.ylabel('Contribution weights')
plt.savefig('WPR_result\Gp' + str(Gpn) +'_Contribution_weights_1' +'.svg', format='svg')
plt.show()

plt.scatter(lon, lat, s = 10,c=contribution_weights, cmap='plasma_r')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar()
plt.title('Contribution weights')
plt.savefig('WPR_result\Gp' + str(Gpn) +'_Contribution_weights_2' +'.svg', format='svg')
plt.show()

# selected_node = similarity.argsort()[:15]
selected_node = node_list[:select_node]    # 倒数最重要的15个
plt.scatter(lon, lat,s = 15, c='gainsboro')
plt.scatter(lon[selected_node], lat[selected_node], s = 20,c='#7e2f8e')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('MGC-RM')
plt.savefig('WPR_result\Gp' + str(Gpn) +'_MGC-RM'+'select'+str(select_node) +'.svg', format='svg')
# plt.savefig(fname="../figure/optimal_distribution_144.svg", format="svg")
plt.show()




