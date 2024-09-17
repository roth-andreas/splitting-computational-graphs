import math
import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
)
from torch_geometric.utils import (
    scatter,
)
import torch_geometric as pyg

from graphgps.layer.dag_utils import init_graph


class DAGCNConv(MessagePassing):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            add_self_loops: bool = True,
            bias: bool = True,
            share_init: bool = False,
            norm: str = 'sym',
            degree_index: int = 0,
            cached: bool = False,
            ordering_type: str = 'degree',
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops

        self.share_init = share_init
        self.norm = norm
        self.degree_index = degree_index
        self.cached = cached
        self.edge_indices = None
        self.edge_weight = None
        self.ordering_type = ordering_type

        self.h_in = in_channels
        self.h_out = out_channels

        self.lins = Linear(self.h_in, self.h_out * 3, bias=False, weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.switch_index = torch.nn.Parameter(torch.tensor([1, 0], dtype=torch.long), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        a = np.sqrt(6 / (self.h_in + self.h_out)) #* torch.nn.init.calculate_gain('relu')
        torch.nn.init.uniform_(self.lins.weight.data, -a, a)
        if self.share_init:
            self.lins.weight = torch.nn.Parameter(torch.cat(
                [self.lins.weight[:self.h_out], self.lins.weight[:self.h_out],
                 self.lins.weight[:self.h_out]], dim=0))

        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_indices: OptTensor = None, edge_weight: OptTensor = None,
                ordering: Tensor = None) -> Tensor:

        if self.cached:
            edge_indices = self.edge_indices
            edge_weight = self.edge_weight
        size = x.size(0)
        if (edge_indices is None) and (ordering is None):
            if self.ordering_type == 'degree':
                ordering = pyg.utils.degree(edge_index[self.degree_index], size, dtype=torch.long)
            elif self.ordering_type == 'random':
                ordering = torch.randperm(size)
            elif self.ordering_type == 'pagerank':
                ordering = pyg.nn.APPNP(K=15, alpha=0.1)(torch.ones((size, 1)), edge_index)

        if edge_weight is None:
            edge_indices_dir, edge_weights_dir = init_graph(x, edge_index,ordering, self.norm, self.add_self_loops, 3)
            if self.cached:
                if self.edge_indices is None:
                    self.edge_indices = edge_indices_dir
                    self.edge_weight = edge_weights_dir
                else:
                    self.edge_indices = edge_indices_dir
                    self.edge_weight = edge_weights_dir
        else:
            edge_indices_dir, edge_weights_dir = edge_indices, edge_weight
        x_dir = x

        x0, x1, x2 = self.lins(x_dir).chunk(3, dim=1)

        xj = torch.cat((x0[edge_indices_dir[0][0]], x1[edge_indices_dir[1][0]], x2[edge_indices_dir[2][0]]),
                       dim=0) * torch.cat(edge_weights_dir, dim=0).unsqueeze(-1)
        index = torch.cat(edge_indices_dir, dim=1)[1]
        x_dir = scatter(xj, index, dim=0, reduce='add', dim_size=size)

        if self.bias is not None:
            x_dir = x_dir + self.bias

        return x_dir
