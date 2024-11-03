from GraphConstruct2 import data_color_graph2
from dataprocess2 import normalization
import numpy as np
import networkx as nx

node = 450
data_num = 2000
adj_ori = np.load('./origin_data/adj_ori'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', allow_pickle=True)
fea_ori = np.load('./origin_data/fea_ori'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', allow_pickle=True)
location_file = np.load('./origin_data/location'+'_node_'+str(node)+'_data_'+str(data_num)+'.npz', allow_pickle=True)
G_o = nx.Graph(adj_ori['arr_0'])
# print('location',location_file.files,location_file['arr_0'].shape)
location = location_file['arr_0']
# print(location)
# f = open('similarity score\zGP_75_node_450_data_2000model_20241102_211834.txt','r')
# a = list(f)
# print(a)
# print(type(a[0]))
# f.close()


score=[]
gpn = 75
error_point = 328
# 标号为数组下标
adjust =  1.463619555579498410e-04#75-329的数据 #1.949527258053421974e-05#159-328# #1.664528177166357636e-04#310-21#
with open('similarity score\gp_159_z_node_450_data_2000model_20241102_165518.txt','r') as f:
# with open('similarity score\gp_310_z_node_450_data_2000model_20241102_165317.txt','r') as f:
# with open('similarity score\zGP_75_node_450_data_2000model_20241102_211834.txt','r') as f:
	for line in f:
		score.append(float(list(line.strip('\n').split(','))[0]))

# print(score)
# print(type(score))
# print(type(score[0]))
# print(score[0])
score_n = normalization(score,error_point,adjust)
print(score_n)

data_color_graph2(score_n,G_o,location,gpn,'rainbow',False)


