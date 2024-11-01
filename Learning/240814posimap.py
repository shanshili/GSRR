import pandas as pd
import matplotlib.pyplot as plt
import NNK
import Modules.Construction.utils as utils
import numpy as np
import networkx as nx

BJ_position = pd.read_csv('../dataset/北京-天津气象数据集2022/北京-天津气象数据集2022/BJ_position.csv')
TJ_position = pd.read_csv('../dataset/北京-天津气象数据集2022/北京-天津气象数据集2022/TJ_position.csv')

dataset_location = pd.concat([BJ_position, TJ_position])
# print(dataset_location)
lat = dataset_location['lat'].values
lon = dataset_location['lon'].values
x = np.transpose(np.vstack((lat, lon)))
#print(x)
print("x.shape:",x.shape)

plt.scatter(lon, lat, 5)  # cmap_name_r, 加_r反转映射关系
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.colorbar()
# plt.savefig('map1.svg', format='svg')
#plt.show()

"""
Same effect as _240830GraphConstruct.py
"""
(K, mask) = NNK.K_nearest_neighbors(x, 'rbf', 4, 2)
print("Kshape:",K.shape)
print("mask:",mask)
print("maskshape:",mask.shape)
adjacency = NNK.knn_graph(K, mask, 4)
#print(adjacency)
#print(np.shape(adjacency))
G = nx.from_numpy_array(adjacency)
# nx.draw(G)
# plt.show()
utils.plot_graph(adjacency, x)

