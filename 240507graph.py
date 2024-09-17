# 导入 networkx 包
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
plt.rcParams.update({
    'figure.figsize': (8, 6)
})


# 创建一个图
g = nx.Graph()
# 添加图的节点
g.add_node(2)
g.add_node(5)
# 添加图的边
g.add_edge(2, 5)
g.add_edge(1, 4)  # 当添加的边对应的节点不存在的时候，会自动创建相应的节点
g.add_edge(1, 2)
g.add_edge(2, 6)
# 绘制图
nx.draw(g,  with_labels=True)
plt.show()
'''
# 默认情况下，networkX 创建的是无向图
G = nx.Graph()
print(G.is_directed())
# 创建有向图
H = nx.DiGraph()
print(H.is_directed())
'''


# 网络平均度的计算
def average_degree(num_edges, num_nodes):
    # this function takes number of edges and number of nodes
    # returns the average node degree of the graph.
    # Round the result to nearest integer (for example 3.3 will be rounded to 3 and 3.7 will be rounded to 4)
    avg_degree = 0
    avg_degree = 2*num_edges/num_nodes
    avg_degree = int(round(avg_degree))
    return avg_degree


# 创建一个空手道俱乐部网络
G2 = nx.karate_club_graph()
# G is an undirected graph
# type(G2)
# 可视化图
# nx.draw(G2,  with_labels=True)
# plt.show()

num_edges = G2.number_of_edges()
num_nodes = G2.number_of_nodes()
avg_degree = average_degree(num_edges, num_nodes)
# print("Average degree of karate club network is {}".format(avg_degree))


# 创建一个二分图 Bipartite Graph
B = nx.Graph()
# Add nodes with the node attribute "bipartite"
B.add_nodes_from([1, 2, 3, 4], bipartite=0)
B.add_nodes_from(["a", "b", "c"], bipartite=1)
# Add edges only between nodes of opposite node sets
B.add_edges_from([(1, "a"), (1, "b"), (2, "b"), (2, "c"), (3, "c"), (4, "a")])
pos = nx.bipartite_layout(B, nx.bipartite.sets(B)[0])  # bipartite position vertical, first part left
nx.draw(B, pos, with_labels=True)   # add position of nodes
# plt.show()


def average_clustering_coefficient(G2):
    """
    this function that takes a nx.Graph
    and returns the average clustering coefficient.
    Round the result to 2 decimal places (for example 3.333 will be rounded to 3.33 and 3.7571 will be rounded to 3.76)
    Note:
    1: Please use the appropriate NetworkX clustering function
    """
    avg_cluster_coef = 0
    avg_cluster_coef = nx.average_clustering(G2)
    avg_cluster_coef = round(avg_cluster_coef, 2)  # result to 2 decimal places
    return avg_cluster_coef


avg_cluster_coef = average_clustering_coefficient(G2)
# print("Average clustering coefficient of karate club network is {}".format(avg_cluster_coef))


print(nx.single_source_shortest_path(g, 5)) # .values() just get the value of dic
print(list(nx.single_source_shortest_path(g, 5).values()))  # tuple to list

def closeness_centrality(G, node=5):
    # the function that calculates closeness centrality
    # for a node in karate club network. G is the input karate club
    # network and node is the node id in the graph. Please round the
    # closeness centrality result to 2 decimal places.

    closeness = 0
    #########################################
    # Raw version following above equation
    # source: https://stackoverflow.com/questions/31764515/find-all-nodes-connected-to-n
    path_length_total = 0
    for path in list(nx.single_source_shortest_path(G, node).values())[1:]:  # Take the first to last element
        path_length_total += len(path)-1  # Subtract self

    closeness = 1 / path_length_total
    closeness = round(closeness, 2)

    return closeness


"""
# Normalized version from NetworkX
# Notice that networkx closeness centrality returns the normalized 
# closeness directly, which is different from the raw (unnormalized) 
# one that we learned in the lecture.
closeness = nx.closeness_centrality(G, node)
print("The karate club network has closeness centrality (normalzied) {:.2f}".format(closeness))
"""

node = 5
closeness = closeness_centrality(g, node=node)
print("The karate club network has closeness centrality (raw) {:.2f}".format(closeness))  # str.format() Format the numbers

