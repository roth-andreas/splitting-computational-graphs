import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym import cfg
import torch_geometric.graphgym.register as register

from graphgps.layer import DA_SAGE


class DASAGEConvLayer(nn.Module):
    """Graph Isomorphism Network with Edge features (GINE) layer.
    """

    def __init__(self, dim_in, dim_out, dropout, residual):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual

        self.act = nn.Sequential(
            register.act_dict[cfg.gnn.act](),
            nn.Dropout(self.dropout),
        )
        self.model = DA_SAGE.DASAGEConv(dim_in, dim_out, bias=True, root_weight=True,
                                          share_init=cfg.gnn.share_init)

    def forward(self, batch):
        x_in = batch.x

        batch.x = self.model(batch.x, batch.edge_index, edge_indices=batch.edge_indices)
        batch.x = self.act(batch.x)

        if self.residual:
            batch.x = x_in + batch.x  # residual connection

        return batch
