import argparse
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.codegen import Print
from sympy.codegen.cnodes import sizeof
from torch.nn.parameter import Parameter
from torch.optim import Adam

import utils
from model import GAT
from evaluation import eva

import sys
sys.path.append("..")
from _240830GraphConstruct import location_graph, topological_features_construct, data_color_graph
from _240810data import get_data2
import matplotlib.pyplot as plt

def pretrain(datasets,A):
    model = GAT(
        num_features=args.input_dim,
        hidden_size=args.hidden_size,
        hidden_size2=args.hidden_size2,
        hidden_size3=args.hidden_size3,
        embedding_size=args.embedding_size,
        alpha=args.alpha,
    ).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    x = torch.tensor(datasets.astype(float)).float()
    adj, adj_label = utils.data_preprocessing(A,location_g,x)
    M = utils.get_M(adj)

    loss_history = []
    for epoch in range(args.max_epoch):
        model.train()
        A_pred, z = model(x, adj, M)
        # print('A_pred{}, z{},A_pred.view(-1){},adj_label.view(-1){},adj_label{}'.format(A_pred, z,A_pred.view(-1).size(),adj_label.view(-1).size(),adj_label))
        loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())  ##
        print('epoch:{}, loss:{}'.format(epoch, loss))
        """
        with torch.no_grad():
            _, z = model(x, adj, M)
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(
                z.data.cpu().numpy()
            )
            acc, nmi, ari, f1 = eva(y, kmeans.labels_, epoch)
        if epoch % 5 == 0:
            torch.save(
                model.state_dict(), f"./pretrain/predaegc_{args.name}_{epoch}.pkl"
            )

        """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(loss_history)), loss_history)
    plt.ylabel('Loss')
    plt.xlabel('Epoch : {}'.format(args.max_epoch))
    plt.title('Training Loss')
    plt.text(0, loss_history[0], loss_history[0])
    plt.text(args.max_epoch, loss_history[(args.max_epoch - 1)], loss_history[(args.max_epoch - 1)],
             horizontalalignment='left')
    plt.savefig('Training_Loss_epoch' + str(args.max_epoch) + '.svg', format='svg')
    plt.show()


    return A_pred, z

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--name", type=str, default="Citeseer")
    parser.add_argument("--max_epoch", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_clusters", default=6, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--hidden_size2", default=2000, type=int)
    parser.add_argument("--hidden_size3", default=400, type=int)
    parser.add_argument("--embedding_size", default=16, type=int)
    parser.add_argument("--weight_decay", type=int, default=5e-3)
    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
    )
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    #datasets = utils.get_dataset(args.name)
    #dataset = datasets[0]
    Path1 = '../dataset/北京-天津气象2021/北京-天津气象2021'
    tem_data, data_location = get_data2(Path1)  # 温度数据作为特征
    location_g, A = location_graph(data_location)
    #dataset = list(location_g)

    if args.name == "Citeseer":
        args.lr = 0.005
        args.k = None
        args.n_clusters = 6
    elif args.name == "Cora":
        args.lr = 0.005
        args.k = None
        args.n_clusters = 7
    elif args.name == "Pubmed":
        args.lr = 0.001
        args.k = None
        args.n_clusters = 3
    else:
        args.k = None

    args.input_dim = tem_data.shape[1]  #特征的大小
    print('args.input_dim',args.input_dim)
    # print(args)
    A_pred, z = pretrain(tem_data,A)
    print(z, np.shape(z))
