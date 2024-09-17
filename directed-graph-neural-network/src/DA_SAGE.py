import math
from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import scatter

from dag_utils import init_graph
import torch_geometric as pyg


class DASAGEConv(MessagePassing):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            share_init: bool = False,
            root_weight: bool = True,
            bias: bool = True,
            degree_index: int = 0,
            cached: bool = False,
            **kwargs,
    ):
        super().__init__('mean', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.root_weight = root_weight
        self.share_init = share_init
        self.degree_index = degree_index
        self.cached = cached
        self.edge_indices = None

        self.h_in = in_channels
        self.h_out = out_channels

        self.lins = Linear(self.h_in, self.h_out * 3, bias=False, weight_initializer='glorot')

        if self.root_weight:
            self.lin_r = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        a = np.sqrt(6 / (self.h_in + self.h_out))
        torch.nn.init.uniform_(self.lins.weight.data, -a, a)
        if self.share_init:
            self.lins.weight = torch.nn.Parameter(torch.cat(
                [self.lins.weight[:self.h_out], self.lins.weight[:self.h_out],
                 self.lins.weight[:self.h_out]], dim=0))
        if self.root_weight:
            self.lin_r.reset_parameters()
            a = np.sqrt(6 / (self.in_channels + self.out_channels))
            torch.nn.init.uniform_(self.lin_r.weight.data, -a, a)

    def forward(
            self,
            x: Tensor,
            edge_index: Adj,
            ordering: Tensor = None,
            edge_indices: OptTensor = None,
    ) -> Tensor:
        if self.cached:
            edge_indices = self.edge_indices
        size = x.size(0)
        if (edge_indices is None) and (ordering is None):
            ordering = pyg.utils.degree(edge_index[self.degree_index], size, dtype=torch.long)

        out = None
        if edge_indices is None:
            edge_indices_dir, _ = init_graph(x, edge_index, ordering, None, False)

            if self.cached:
                if self.edge_indices is None:
                    self.edge_indices = [edge_indices_dir]
                else:
                    self.edge_indices.append(edge_indices_dir)
        else:
            edge_indices_dir = edge_indices
        x_dir = x

        x0, x1, x2 = self.lins(x_dir).chunk(3, dim=1)

        index = torch.cat(edge_indices_dir, dim=1)[1]
        # propagate_type: (x: OptPairTensor)
        x_j = torch.cat((x0[edge_indices_dir[0][0]],
                         x1[edge_indices_dir[1][0]],
                         x2[edge_indices_dir[2][0]]
                         ), dim=0)
        x_dir = scatter(x_j, index, dim=0, reduce='mean', dim_size=size)
        if out is None:
            out = x_dir
        else:
            out += x_dir

        if self.root_weight:
            out = out + self.lin_r(x)

        return out
