import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch import nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from graphgps.layer.da_gcn_conv_layer import DAGCNConvLayer
from graphgps.layer.da_sage_conv_layer import DASAGEConvLayer
from graphgps.layer.dag_utils import init_graph
from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvLayer
from graphgps.layer.gcn_conv_layer import GCNConvLayer
from graphgps.layer.sage_conv_layer import SAGEConvLayer
import torch_geometric as pyg


@register_network('custom_gnn')
class CustomGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.model_type = cfg.gnn.layer_type
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        conv_model = self.build_conv_model(cfg.gnn.layer_type)
        self.gnn_layers = torch.nn.ModuleList()
        for _ in range(cfg.gnn.layers_mp):
            self.gnn_layers.append(conv_model(dim_in,
                                              dim_in,
                                              dropout=cfg.gnn.dropout,
                                              residual=cfg.gnn.residual))
        GNNHead = register.head_dict[cfg.gnn.head]
        if cfg.gnn.jk is not None:
            self.jk = JumpingKnowledge(cfg.gnn.jk)
            jk_dim = cfg.gnn.dim_inner if cfg.gnn.jk == 'max' else cfg.gnn.dim_inner * cfg.gnn.layers_mp
            self.post_mp = GNNHead(dim_in=jk_dim, dim_out=dim_out)
        else:
            self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

        self.act = nn.Sequential(
            register.act_dict[cfg.gnn.act](),
            nn.Dropout(cfg.gnn.dropout),
        )
        self.ordering_type = cfg.gnn.ordering_type
        if self.ordering_type == 'pagerank':
            self.ppr = pyg.nn.APPNP(K=15, alpha=0.1)

    def build_conv_model(self, model_type):
        if model_type == 'gatedgcnconv':
            return GatedGCNLayer
        elif model_type == 'gineconv':
            return GINEConvLayer
        elif model_type == 'gcnconv':
            return GCNConvLayer
        elif model_type == 'da-gcnconv':
            return DAGCNConvLayer
        elif model_type == 'sageconv':
            return SAGEConvLayer
        elif model_type == 'da-sageconv':
            return DASAGEConvLayer
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        if cfg.gnn.layer_type.startswith('da'):
            size = batch.x.size(0)
            if self.ordering_type == 'degree':
                ordering = pyg.utils.degree(batch.edge_index[0], size, dtype=torch.long)
            elif self.ordering_type == 'random':
                ordering = batch.ordering
            elif self.ordering_type == 'pagerank':
                ordering = self.ppr(torch.ones((size, 1),device=batch.x.device), batch.edge_index).squeeze()
            elif self.ordering_type == 'features':
                ordering = torch.sum(batch.x, dim=1)
            batch.edge_indices, batch.edge_weight = init_graph(batch.x,
                                                               batch.edge_index,
                                                               ordering,
                                                               'sym' if cfg.gnn.layer_type == 'da-gcnconv' else None, False,
                                                               3)
        elif cfg.gnn.layer_type == 'gcnconv':
            batch.edge_weight = gcn_norm(batch.edge_index, None, batch.x.size(0), False, False, 'source_to_target',
                                         torch.float32)[1]

        batch = self.encoder(batch)
        if cfg.gnn.layers_pre_mp > 0:
            batch = self.pre_mp(batch)
        xs = []
        for idx, conv in enumerate(self.gnn_layers):
            batch = conv(batch)
            xs.append(batch.x)
        if cfg.gnn.jk:
            batch.x = self.jk(xs)

        return self.post_mp(batch)
