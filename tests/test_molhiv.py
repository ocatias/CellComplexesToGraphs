import torch

from data.dataset import CellEncodedDataset
DATA_PATH = r'./datasets'

dataset = CellEncodedDataset(DATA_PATH, "ogbg-molhiv", 0, False, False, True, True)
from data.graph_operations import get_rings, ring_enc_with_id_maps
from experiments.visualizations import visualize_from_edge_index


for i in range(len(dataset)):
    print("\r", i, end="")

    data = dataset[i]

    data_enc, id_maps = ring_enc_with_id_maps(data, 7, False, True, True)

    for j in range(data.edge_index.shape[1]):
        assert torch.equal(data.edge_index[:,j], data_enc.edge_index[:,j])
        assert torch.equal(data.edge_attr[j,:], data_enc.edge_attr[j,:])

visualize_from_edge_index(data.edge_index)

visualize_from_edge_index(data_enc.edge_index, id_maps)
