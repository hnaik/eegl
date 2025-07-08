import logging

import torch
import torch.nn.functional as F

from torch_geometric.nn import GraphConv

LOG = logging.getLogger(__name__)


class GCN(torch.nn.Module):
    def __init__(self, dataset, params):
        super().__init__()
        self.dropout = params['dropout']
        input_dim = dataset.num_node_features
        hidden_dim = params['hidden_dim']
        embedding_dim = hidden_dim * 2

        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, embedding_dim)
        self.conv3 = GraphConv(embedding_dim, dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)
