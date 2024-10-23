import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channel, hid_channel, out_channel):
        super(GCNEncoder, self).__init__()

        self.conv1 = GCNConv(in_channel, hid_channel)
        self.conv_mu = GCNConv(hid_channel, out_channel)
        self.conv_logstd = GCNConv(hid_channel, out_channel)

    def forward(self, x, edge_index):
        # print(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)



