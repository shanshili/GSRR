import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        # torch.nn.Parameter()将一个不可训练的tensor转换成可以训练的类型parameter，
        # 并将这个parameter绑定到这个module里面。
        # 即在定义网络时这个tensor就是一个可以训练的参数了。
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # print('self.W',self.W)  # 在变化

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.a = nn.Linear(754,1)

    def forward(self, input, adj, M, concat=True):
        h = torch.mm(input, self.W)  #特征增维
        # print('self.W_', self.W)
        # print('h',h)
        # print('h', h.size())
        # e = self._prepare_attentional_mechanism_input(h)
        # print('e', e)
        #
        # zero_vec = -9e15 * torch.ones_like(e)
        # # 将没有连接的边置为负无穷
        # attention = torch.where(adj > 0, e, zero_vec)
        # # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        # attention = F.softmax(attention, dim=1)
        # # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        # attention = F.dropout(attention, self.dropout, training=self.training)
        # # dropout，防止过拟合
        # h_prime = torch.matmul(attention, h)
        # # 得到由周围节点通过注意力权重进行更新的表示
        # # print('h_prime', h_prime)
        #分别与A相乘再拼接
        attn_for_self = torch.mm(h, self.a_self)  # (N,1)
        attn_for_neighs = torch.mm(h, self.a_neighs)  # (N,1)
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs, 0, 1)

        # attn_dense = torch.mul(attn_dense, M) ####？？？
        e = self.a(attn_dense)
        attn_dense = self.leakyrelu(e)  # (N,N)

        zero_vec = -9e15 * torch.ones_like(adj)
        adj = torch.where(adj > 0, attn_dense, zero_vec)
        attention = F.softmax(adj, dim=1)
        # attention = F.softmax(attn_dense, dim=1)
        h_prime = torch.matmul(attention, h)


        if concat:
            return F.elu(h_prime)
        else:
            return h_prime

    # def _prepare_attentional_mechanism_input(self, Wh):
    #     # Wh.shape (N, out_feature)
    #     # self.a.shape (2 * out_feature, 1)
    #     # Wh1&2.shape (N, 1)
    #     # e.shape (N, N)
    #     Wh1 = torch.matmul(Wh, self.a_self[:self.out_features, :])
    #     Wh2 = torch.matmul(Wh, self.a_self[self.out_features:, :])
    #     # broadcast add
    #     e = Wh1 + Wh2.T
    #     return self.leakyrelu(e)

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )