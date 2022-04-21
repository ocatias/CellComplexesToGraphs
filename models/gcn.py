import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, normalize=False, dropout = 0.1,
            hidden_dim = 64, nr_layers = 2):

        super().__init__()
        if nr_layers < 1:
            raise ValueError("nr_layers must be at least 1")

        self.conv_layers = nn.ModuleList([GCNConv(num_features, hidden_dim)])
        for i in range(nr_layers - 1):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))

        self.linear = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        for layer in self.conv_layers:
            x = F.relu(layer(x, edge_index))
            x = self.dropout(x)

        x = global_mean_pool(x, data.batch)
        x = self.linear(x)
        return F.softmax(x, dim = 1)
