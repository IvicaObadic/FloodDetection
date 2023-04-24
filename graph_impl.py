import torch
from torch import Tensor
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F


class FloodDetectionGraph(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        #self.in_conv = nn.Conv1d(in_channels, hidden_channels)
        self.conv_message_passing = GCNConv(in_channels, hidden_channels)
        self.nonlinear_fn = nn.ReLU()
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv_message_passing(x, edge_index)
        x = self.nonlinear_fn(x)
        # x = self.conv_message_passing_2(x, edge_index)
        # x = self.nonlinear_fn(x)
        # x = self.conv_message_passing_3(x, edge_index)
        # x = self.nonlinear_fn(x)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.2, training=self.training)
        y = self.linear(x)

        return y

