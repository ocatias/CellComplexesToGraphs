import torch
from torch_geometric.data import Data

from data.graph_operations import simplex_encoding

DATA_PATH = r'./datasets'


edge_index = torch.tensor([[0, 1, 1, 2, 0, 2],
                           [1, 0, 2, 1, 2, 0]], dtype=torch.long)
x = torch.tensor([[3], [4], [5]], dtype=torch.float)
edge_features = torch.tensor([[10], [10], [100], [100], [1000], [1000]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index, edge_attr=edge_features)
print(data)

data2 = simplex_encoding(data, 3, True, True)
print(data2)
print(data2.x)
print(data2.edge_index)


from ogb.graphproppred import PygGraphPropPredDataset

dataset = PygGraphPropPredDataset(root=DATA_PATH, name = "ogbg-molhiv")
nr_graphs = len(dataset)
s = 0
for i in range(len(dataset)):
    enc1 = simplex_encoding(dataset[i], 5, True, True)
    enc2 = simplex_encoding(dataset[i], 2, True, True)

    delta = enc1.x.shape[0]- enc2.x.shape[0]
    if i % 100 == 0:
        print(f"Cliques: {s}, graphs: {i} / {nr_graphs}")
    s += delta

print("Cliques:", s)

