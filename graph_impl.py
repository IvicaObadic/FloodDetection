import torch
from torch import Tensor
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool, global_mean_pool, SAGPooling


import torch.nn.functional as F


class FloodDetectionGraph(torch.nn.Module):
    def __init__(self, num_gcn_layers, in_channels, hidden_channels, out_channels, pooling_layer):
        super().__init__()
        self.gcn_conv_layers = nn.ModuleList([GCNConv(in_channels, hidden_channels)])
        for i in range(1, num_gcn_layers):
            self.gcn_conv_layers.append(GCNConv(hidden_channels, hidden_channels))

        self.nonlinear_fn = nn.ReLU()
        self.pooling_layer = pooling_layer
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch) -> Tensor:
        #print(x.shape)
        for i, gcn_layer in enumerate(self.gcn_conv_layers):
            x = gcn_layer(x, edge_index)
            x = self.nonlinear_fn(x)

        if isinstance(self.pooling_layer, SAGPooling):
            sag_output = self.pooling_layer(x, edge_index, batch=batch)
            x = sag_output[0]
            edge_index = sag_output[1]
            batch= sag_output[3]
            x = global_mean_pool(x, batch = batch)
        else:
            x = self.pooling_layer(x, batch=batch)
        #print(x.shape)
        #x = F.dropout(x, p=0.2, training=self.training)
        y = self.linear(x)

        return y

