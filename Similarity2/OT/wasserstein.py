import numpy as np
import pandas as pd
import torch

def wasserstein(e_1, e_2, s_1, s_2):
    e_1 = torch.tensor(e_1)
    e_2 = torch.tensor(e_2)
    s_1 = torch.tensor(s_1)
    s_2 = torch.tensor(s_2)

    c_1 = torch.mm(s_1, s_1.t())
    c_2 = torch.mm(s_2, s_2.t())

    p_1 = torch.sum(torch.pow((e_1 - e_2), 2), 1)
    p_2 = torch.sum(torch.pow(torch.pow(c_1, 1/2) - torch.pow(c_2, 1/2), 2), 1)
    return torch.sum(p_1 + p_2)

# mu = pd.read_csv('mu.csv', index_col=0).values
# std = pd.read_csv('std.csv', index_col=0).values
mu = np.load('mu.npz')['arr_0']
std = np.load('std.npz')['arr_0']


w_total = []
for i in range(1, 145):
    w = wasserstein(mu[0], mu[i], std[0], std[i])
    w_total.append(w)

# print(len(w_total))
# print(w_total)
# w_total = np.array(w_total)
selected_w, selected_node = torch.sort(torch.tensor(w_total), descending=True)
print(selected_node)
print(w_total)
