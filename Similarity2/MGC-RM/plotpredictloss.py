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
location = location_file['arr_0']



Loss=[]
gpn = '75_loss'
error_point = 328
# 标号为数组下标
adjust =  0
with open('prediction_loss\GraphPair_75_n450_d2000_Prediction_Loss_epoch_21_lr_0.0001_20241102_211834.txt','r') as f:
	for line in f:
		Loss.append(float(list(line.strip('\n').split(','))[0]))

# print(score)
# print(type(score))
# print(type(score[0]))
# print(score[0])

# score_n = normalization(Loss,error_point,adjust)
# print(Loss)
data_color_graph2(Loss,G_o,location,gpn,'Reds',True)


