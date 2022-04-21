import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform

from conv import GNN_node, GNN_node_Virtualnode

from torch_scatter import scatter_mean

class GNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean", dim_pooling=False):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

        self.dim_pooling = dim_pooling
        if self.dim_pooling:
            self.lin_per_dim = torch.nn.ModuleList([torch.nn.Linear(self.emb_dim, self.emb_dim) for _ in range(3)])

        self.non_linearity = torch.nn.ReLU()

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)


        if self.dim_pooling:
            dimensional_pooling = []
            for dim in range(3):
                multiplier = torch.unsqueeze(batched_data.vertex_dim[:, dim], dim=1)
                single_dim = h_node * multiplier

                single_dim = self.pool(single_dim, batched_data.batch)
                
                if self.graph_pooling == "mean":   
                    # Scale up the number of nodes, since we are overcounting nodes whose vectors we have set to 0
                    single_dim = single_dim / torch.sum(batched_data.vertex_dim[:, dim]) * batched_data.x.shape[0]

                single_dim = self.non_linearity(self.lin_per_dim[dim](single_dim))

                dimensional_pooling.append(single_dim)
            h_graph = sum(dimensional_pooling)
        else:
            h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)


if __name__ == '__main__':
    GNN(num_tasks = 10)