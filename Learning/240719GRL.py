"""
graph represents learning
"""
import networkx as nx
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.optim import SGD
import random


# 载入空手道俱乐部网络
G = nx.karate_club_graph()
# 可视化图
# nx.draw(G, with_labels=True)

torch.manual_seed(1)

"""
torch.nn.Embedding是用来将一个数字变成一个指定维度的向量的，
比如数字1变成一个128维的向量，数字2变成另外一个128维的向量。
不过，这128维的向量并不是永恒不变的，这些128维的向量是模型真正的输入（也就是模型的第1层）（数字1和2并不是，可以算作模型第0层），
然后这128维的向量会参与模型训练并且得到更新，从而数字1会有一个更好的128维向量的表示

https://blog.csdn.net/qq_43391414/article/details/120783887
one-hot向量没有任何的语义信息,需要一个低维稠密的向量来代替one-hot向量

embedding=torch.nn.Embedding(vocab_size=2,emb_size=2)
#vocab_size：表示一共有多少个字需要embedding，
#emb_size:表示我们希望一个字向量的维度是多少。

Embedding和Linear几乎是一样的，区别就在于：输入不同，一个是输入数字，后者是输入one-hot向量
"""
"""
torch.rand(*sizes, out=None) → Tensor
返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义。

Embedding这个类有个属性weight，它是torch.nn.parameter.Parameter类型的，
作用就是存储真正的word embeddings。如果不给weight赋值，Embedding类会自动给他初始化，
看上述代码第6~8行，如果属性weight没有手动赋值，则会定义一个torch.nn.parameter.Parameter对象，
然后对该对象进行reset_parameters()，看第21行，对self.weight先转为Tensor在对其进行normal_(0, 1)
(调整为$N(0, 1)$正态分布)。所以nn.Embeddig.weight默认初始化方式就是N(0, 1)分布，即均值$\mu=0$，方差$\sigma=1$的标准正态分布
"""


# 初始化嵌入函数
def create_node_emb(num_node=34, embedding_dim=16):
    emb = nn.Embedding(num_node, embedding_dim)  # 创建 Embedding
    emb.weight.data = torch.rand(num_node, embedding_dim)  # 均匀初始化                                          ?????????
    return emb


"""
随机初始化嵌入：
我们希望空手道俱乐部网络中的每个节点都有 16 维向量。
我们要初始化均匀分布的矩阵，范围为 [0,1)，使用 torch.rand。
可视化嵌入：将Embedding用PCA降维到二维，再将两类节点的嵌入的二维表示分别以红色和蓝色画出点。
"""
# 初始化嵌入
emb = create_node_emb()
#print(emb.weight)

"""
主成分分析（Principal Components Analysis），简称PCA，是一种数据降维技术，用于数据预处理。
n_components:  PCA算法中所要保留的主成分个数n，也即保留下来的特征个数n
fit_transform(X):用X来训练PCA模型，同时返回降维后的数据。
"""


# 可视化
def visualize_emb(emb):
    X = emb.weight.data.numpy()
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)  # 根据初始化的嵌入函数的权重参数，训练PCA，对图进行可视化表示
    #print(components)
    plt.figure(figsize=(6, 6))  # Create a new figure, or activate an existing figure.
    club1_x = []
    club1_y = []
    club2_x = []
    club2_y = []
    #print(G.nodes)
    """
    Graph.nodes.data
        string or bool, optional (default=False)
        The node attribute returned in 2-tuple (n, ddict[data]). 
    Graph.nodes just return node 
    
    G.nodes(data=True):  A NodeView of the Graph as G.nodes or G.nodes().
    """
    for node in G.nodes(data=True):  # A NodeView of the Graph as G.nodes or G.nodes().
        if node[1]['club'] == 'Mr. Hi':
            """
            node的形式：第一个元素是索引，第二个元素是attributes字典
            node[1]、node[0]、node[1]['club']的区别   ？？？？？？？
            """
            #print(node[1])
            #print(components[node[0]][0])
            club1_x.append(components[node[0]][0])
            club1_y.append(components[node[0]][1])
            """ 
            这里添加的元素就是节点对应的embedding经PCA后的两个维度
            """

        else:
            club2_x.append(components[node[0]][0])
            club2_y.append(components[node[0]][1])
    plt.scatter(club1_x, club1_y, color="red", label="Mr. Hi")
    plt.scatter(club2_x, club2_y, color="blue", label="Officer")
    plt.legend()  # add legend
    plt.show()


# 可视化初始嵌入
visualize_emb(emb)



"""
我们将使用边分类为正或负的任务来完成表示学习。
获取负边和正边。正边是图中存在的边，存放在 pos_edge_list 中。
"""


def graph_to_edge_list(G):
    # 将 tensor 变成 edge_list
    edge_list = []
    #print(G.edges())
    for edge in G.edges():
        edge_list.append(edge)
    #print(edge_list)
    return edge_list


def edge_list_to_tensor(edge_list):
    # 将 edge_list 变成 tensor
    edge_index = torch.tensor([])
    edge_index = torch.LongTensor(edge_list).t()
    # print(edge_index)
    """
    torch.LongTensor是64位整型
    torch.tensor是一个类，用于生成一个单精度浮点类型的张量。
    
    torch.Tensor is a multi-dimensional matrix containing elements of a single data type.
    torch.LongTensor(edge_list).t()    Returns a view of this tensor with its dimensions reversed.
    """
    return edge_index


pos_edge_list = graph_to_edge_list(G)
pos_edge_index = edge_list_to_tensor(pos_edge_list)
print("The pos_edge_index tensor has shape {}".format(pos_edge_index.shape))
print("The pos_edge_index tensor has sum value {}".format(torch.sum(pos_edge_index)))
"""
torch.sum Returns the sum of all elements in the input tensor.
"""

"""
负边：图中不存在的边，即两个节点之间在真实图中没有连线的边。抽样一定数目不存在的边作为负值的边。
"""

"""
enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
random.sample 多用于截取列表的指定长度的随机数，但是不会改变列表本身的排序
"""


# 采样负边
def sample_negative_edges(G, num_neg_samples):
    neg_edge_list = []
    # 得到图中所有不存在的边（这个函数只会返回一侧，不会出现逆边）
    non_edges_one_side = list(enumerate(nx.non_edges(G)))
    neg_edge_list_indices = random.sample(range(0,len(non_edges_one_side)), num_neg_samples)

    # 取样num_neg_samples长度的索引
    for i in neg_edge_list_indices:
        neg_edge_list.append(non_edges_one_side[i][1])
    return neg_edge_list


# Sample 78 negative edges
neg_edge_list = sample_negative_edges(G, len(pos_edge_list))

# Transform the negative edge list to tensor
neg_edge_index = edge_list_to_tensor(neg_edge_list)
print("The neg_edge_index tensor has shape {}".format(neg_edge_index.shape))
"""
为什么要列表转向量？？？？？？
"""


"""
给定一条边和每个节点的嵌入，嵌入的点积，后跟一个 sigmoid，应该给出该边为正（sigmoid 的输出 > 0.5）
或负（sigmoid 的输出 < 0.5）的可能性。训练目标：使有边连接（pos_edge_index）的节点嵌入点乘结果趋近于1，无边连接的趋近于0。
"""


def accuracy(pred, label):
    # 题目要求：
    # 输入参数：
    #  pred (the resulting tensor after sigmoid)
    #  label (torch.LongTensor)
    # 预测值大于0.5被分类为1，否则就为0
    # 准确率返回值保留4位小数

    # accuracy=预测与实际一致的结果数/所有结果数
    # pred和label都是[78*2=156]大小的Tensor
    accu = round(((pred > 0.5) == label).sum().item() / (pred.shape[0]), 4)
    return accu


def train(emb, loss_fn, sigmoid, train_label, train_edge):
    # 题目要求：
    # 用train_edge中的节点获取节点嵌入
    # 点乘每一点对的嵌入，将结果输入sigmoid
    # 将sigmoid输出输入loss_fn
    # 打印每一轮的loss和accuracy

    epochs = 500
    learning_rate = 0.1

    optimizer = SGD(emb.parameters(), lr=learning_rate, momentum=0.9)

    for i in range(epochs):
        optimizer.zero_grad()
        train_node_emb = emb(train_edge)  # [2,156,16]
        # 156是总的用于训练的边数，指78个正边+78个负边
        dot_product_result = train_node_emb[0].mul(train_node_emb[1])  # 点对之间对应位置嵌入相乘，[156,16]
        dot_product_result = torch.sum(dot_product_result, 1)  # 加起来，构成点对之间向量的点积，[156]
        sigmoid_result = sigmoid(dot_product_result)  # 将这个点积结果经过激活函数映射到0,1之间
        loss_result = loss_fn(sigmoid_result, train_label)
        loss_result.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f'loss_result {loss_result}')
            print(f'Accuracy {accuracy(sigmoid_result, train_label)}')


loss_fn = nn.BCELoss()
sigmoid = nn.Sigmoid()

# 生成正负样本标签
pos_label = torch.ones(pos_edge_index.shape[1], )
neg_label = torch.zeros(neg_edge_index.shape[1], )

# 拼接正负样本标签
train_label = torch.cat([pos_label, neg_label], dim=0)

# 拼接正负样本
# 因为数据集太小，我们就全部作为训练集
train_edge = torch.cat([pos_edge_index, neg_edge_index], dim=1)

train(emb, loss_fn, sigmoid, train_label, train_edge)

visualize_emb(emb)


