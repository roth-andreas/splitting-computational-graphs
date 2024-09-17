import torch

import DA_GCN
import DA_SAGE
from torch_geometric.utils import degree
import torch.nn as nn
import torch_geometric as pyg


def dirichlet_energy(x, edge_index, rw=False):
    with torch.no_grad():
        src, dst = edge_index
        deg = degree(src, num_nodes=x.shape[0])
        x = x / torch.norm(x)
        if not rw:
            x = x / torch.sqrt(deg + 0.0).view(-1, 1)
        energy = torch.norm(x[src] - x[dst], dim=1, p=2) ** 2.0

        energy = energy.mean()

        energy *= 0.5

    return float(energy.mean().detach().cpu())

def rank_diff(x):
    with torch.no_grad():
        x = x / torch.linalg.norm(x, 'nuc')

        i = x.abs().sum(dim=1).argmax()
        j = x.abs().sum(dim=0).argmax()
        mean0 = x[i].view(1, -1)
        mean1 = x[:, j].view(-1, 1)
        if mean0[0,j] < 0:
            mean0 = -mean0
        x_hat = mean1 @ mean0
        x_hat = x_hat / torch.linalg.norm(x_hat, 'nuc')

        return torch.linalg.norm(x - x_hat, 'nuc').item()


class SimpleModel(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim, num_layers, conv):
        super().__init__()
        self.enc = nn.Linear(in_dim, h_dim)
        self.conv_type = conv
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            dim2 = h_dim if i < num_layers - 1 else out_dim
            if conv == 'GCN':
                layer = pyg.nn.GCNConv(h_dim, dim2, bias=True, add_self_loops=False)
            elif conv == 'SAGE':
                layer = pyg.nn.SAGEConv(h_dim, dim2, bias=True)
            elif conv == 'MRS-GCN':
                layer = DA_GCN.DAGCNConv(h_dim, dim2, bias=True, add_self_loops=False)
            elif conv == 'MRS-SAGE':
                layer = DA_SAGE.DASAGEConv(h_dim, dim2, bias=True)
            self.convs.append(layer)

        self.num_layers = num_layers

    def forward(self, data):
        edge_index = data.edge_index

        stats = []

        x = self.enc(data.x.float())
        stats.append(rank_diff(x))
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i < self.num_layers - 1:
                x = torch.relu(x)
            stats.append(rank_diff(x))

        return x, stats
