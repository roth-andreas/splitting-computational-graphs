import os

import gdown
import numpy as np
import torch
from scipy.io import loadmat
from torch_geometric.data import Data
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch_geometric as pyg

from torch_sparse import mul
from torch_sparse import sum as sparsesum
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.nn.conv.gcn_conv import gcn_norm

def directed_norm_adj(adj, norm_adj):
    """
    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
    """
    in_deg = sparsesum(norm_adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

    out_deg = sparsesum(norm_adj, dim=1)
    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

    adj = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, in_deg_inv_sqrt.view(1, -1))
    return adj


def directed_norm(edge_index, num_nodes=None, dtype=None):
    """
    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    in_deg = scatter(edge_weight, col, dim=0, dim_size=num_nodes, reduce='sum')
    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float('inf'), 0.0)

    out_deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes, reduce='sum')
    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float('inf'), 0.0)

    edge_weight = in_deg_inv_sqrt[col] * edge_weight * out_deg_inv_sqrt[row]

    return edge_index, edge_weight


@torch.no_grad()
def init_graph(x, edge_index, ordering, norm='sym', add_self_loops=False):
    num_nodes = x.shape[0]

    if norm is None:
        edge_ind_cur, _ = split_graph(edge_index, x.size(0), ordering)
        return edge_ind_cur, None

    if norm == 'dir':
        ed_in, ed_we = directed_norm(edge_index, num_nodes)
    elif norm == 'sym':
        ed_in, ed_we = gcn_norm(edge_index, None, num_nodes, False, add_self_loops,
                                'source_to_target', torch.float32)
    edge_ind_cur, indices_list = split_graph(ed_in, x.size(0), ordering)
    return edge_ind_cur, [ed_we[indices_list[j]] for j in range(len(edge_ind_cur))]


def split_graph(edge_index, num_nodes, ordering=None, ordering_type='degree'):
    if ordering is None:
        if ordering_type == 'degree':
            ordering = pyg.utils.degree(edge_index[0], num_nodes, dtype=torch.long)
        elif ordering_type == 'random':
            ordering = torch.randperm(num_nodes)
        elif ordering_type == 'pagerank':
            ordering = pyg.nn.APPNP(K=15, alpha=0.1)(torch.ones((num_nodes, 1)), edge_index)
    indices_list = []
    edge_indices = []
    odering_start = ordering[edge_index[0]]
    odering_end = ordering[edge_index[1]]
    indices = odering_start < odering_end
    indices_list.append(indices)
    edge_set = edge_index[:, indices]
    edge_indices.append(edge_set)
    indices = odering_start > odering_end
    indices_list.append(indices)
    edge_set = edge_index[:, indices]
    edge_indices.append(edge_set)
    indices = odering_start == odering_end
    indices_list.append(indices)
    edge_set = edge_index[:, indices]
    edge_indices.append(edge_set)

    return edge_indices, indices_list