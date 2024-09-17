import os

import gdown
import numpy as np
import torch_geometric as pyg
import torch
import networkx as nx
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


def get_mask(idx, num_nodes):
    """
    Given a tensor of ids and a number of nodes, return a boolean mask of size num_nodes which is set to True at indices
    in `idx`, and to False for other indices.
    """
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask

# adapting
# https://github.com/CUAI/Non-Homophily-Large-Scale/blob/82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6/data_utils.py#L221
# load splits from here https://github.com/CUAI/Non-Homophily-Large-Scale/tree/82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6/data/splits
def process_fixed_splits(splits_lst, num_nodes):
    n_splits = len(splits_lst)
    train_mask = torch.zeros(num_nodes, n_splits, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, n_splits, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, n_splits, dtype=torch.bool)
    for i in range(n_splits):
        train_mask[splits_lst[i]["train"], i] = 1
        val_mask[splits_lst[i]["valid"], i] = 1
        test_mask[splits_lst[i]["test"], i] = 1
    return train_mask, val_mask, test_mask

# adapting - https://github.com/CUAI/Non-Homophily-Large-Scale/blob/82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6/dataset.py#L257
def load_snap_patents_mat(n_classes=5, root="dataset/"):
    dataset_drive_url = {"snap-patents": "1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia"}
    splits_drive_url = {"snap-patents": "12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N"}

    # Build dataset folder
    if not os.path.exists(f"{root}snap_patents"):
        os.mkdir(f"{root}snap_patents")

    # Download the data
    if not os.path.exists(f"{root}snap_patents/snap_patents.mat"):
        p = dataset_drive_url["snap-patents"]
        print(f"Snap patents url: {p}")
        gdown.download(
            id=dataset_drive_url["snap-patents"],
            output=f"{root}snap_patents/snap_patents.mat",
            quiet=False,
        )

    # Get data
    fulldata = loadmat(f"{root}snap_patents/snap_patents.mat")
    edge_index = torch.tensor(fulldata["edge_index"], dtype=torch.long)
    node_feat = torch.tensor(fulldata["node_feat"].todense(), dtype=torch.float)
    num_nodes = int(fulldata["num_nodes"])
    years = fulldata["years"].flatten()
    label = even_quantile_labels(years, n_classes, verbose=False)
    label = torch.tensor(label, dtype=torch.long)

    # Download splits
    name = "snap-patents"
    if not os.path.exists(f"{root}snap_patents/{name}-splits.npy"):
        assert name in splits_drive_url.keys()
        gdown.download(
            id=splits_drive_url[name],
            output=f"{root}snap_patents/{name}-splits.npy",
            quiet=False,
        )

    # Get splits
    splits_lst = np.load(f"{root}snap_patents/{name}-splits.npy", allow_pickle=True)
    train_mask, val_mask, test_mask = process_fixed_splits(splits_lst, num_nodes)
    data = Data(
        x=node_feat,
        edge_index=edge_index,
        y=label,
        num_nodes=num_nodes,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    dataset = [data]#DummyDataset(data, n_classes)

    return dataset


class DummyDataset(object):
    def __init__(self, data, num_classes):
        self._data = data
        self.num_classes = num_classes



# Taken verbatim from https://github.com/CUAI/Non-Homophily-Large-Scale/blob/82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6/data_utils.py#L39
def even_quantile_labels(vals, nclasses, verbose=True):
    """partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on

    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int64)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print("Class Label Intervals:")
        for class_idx, interval in enumerate(interval_lst):
            print(f"Class {class_idx}: [{interval[0]}, {interval[1]})]")
    return label

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
def init_graph(x, edge_index, ordering, norm='sym', add_self_loops=False, num_graphs=3):
    num_nodes = x.shape[0]

    if norm is None:
        edge_ind_cur, _ = split_graph(edge_index, num_graphs, x.size(0), ordering)
        return edge_ind_cur, None

    if norm == 'dir':
        ed_in, ed_we = directed_norm(edge_index, num_nodes)
    elif norm == 'sym':
        ed_in, ed_we = gcn_norm(edge_index, None, num_nodes, False, add_self_loops,
                                'source_to_target', torch.float32)
    edge_ind_cur, indices_list = split_graph(ed_in, num_graphs, x.size(0), ordering)
    return edge_ind_cur, [ed_we[indices_list[j]] for j in range(len(edge_ind_cur))]


def split_graph(edge_index, num_graphs, num_nodes, ordering=None, ordering_type='degree'):
    if ordering is None:
        if ordering_type == 'degree':
            ordering = pyg.utils.degree(edge_index[0], num_nodes, dtype=torch.long)
        elif ordering_type == 'random':
            ordering = torch.randperm(num_nodes)
        elif ordering_type == 'pagerank':
            ordering = pyg.nn.APPNP(K=15, alpha=0.1)(torch.ones((num_nodes, 1)), edge_index)
    indices_list = []
    edge_indices = []
    if num_graphs == 1:
        edge_indices.append(edge_index)
        indices_list.append(torch.ones(edge_index.size(1), dtype=torch.bool))
    elif num_graphs >= 3:
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
    if num_graphs in [-3,6]:
        ordering = pyg.utils.degree(edge_index[1], num_nodes, dtype=torch.long)
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