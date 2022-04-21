import os

import torch_geometric

from SPE.ring_encoding import CellularRingEncoding
from experiments.visualizations import visualize_from_edge_index

DATA_PATH = r'./datasets'

ring_enc1 = CellularRingEncoding(3, True, True, True, True)
ring_enc2 = CellularRingEncoding(10, False, True, True, True)


ds0 = torch_geometric.datasets.TUDataset(os.path.join(DATA_PATH), name="MUTAG", use_node_attr=True, use_edge_attr=True)
print("ds0", ds0)

ds1= torch_geometric.datasets.TUDataset(os.path.join(DATA_PATH, repr(ring_enc1)), name="MUTAG", pre_transform=ring_enc1, use_node_attr=True, use_edge_attr=True)
print("DS1", ds1)
ds2= torch_geometric.datasets.TUDataset(os.path.join(DATA_PATH, repr(ring_enc2)), name="MUTAG", pre_transform=ring_enc2, use_node_attr=True, use_edge_attr=True)
print("DS2", ds2)

ds3= torch_geometric.datasets.TUDataset(os.path.join(DATA_PATH, repr(ring_enc1)), name="MUTAG", pre_transform=ring_enc1, use_node_attr=True, use_edge_attr=True)
print("DS3", ds3)


visualize_from_edge_index(ds0[0].edge_index)

data, id_maps = ring_enc2.encode_with_id_maps(ds0[0])
visualize_from_edge_index(data.edge_index, id_maps)
