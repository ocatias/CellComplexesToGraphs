import os
import pickle
import glob

import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, download_url
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
from tqdm import tqdm
from torch_geometric.data.makedirs import makedirs

from data.graph_operations import simplex_encoding, ring_encoding

implemented_TU_datasets = ["MUTAG", "PROTEINS", "IMDB-BINARY", "COLLAB", "NCI1", "NCI109"]

# Datasets that are supposed to use a train, validation and test split
tvt_datasets = ["MOLHIV"]


class CellEncodedDataset(InMemoryDataset):
    def __init__(self, root, dataset, max_struct_dim, use_cliques = False, edge_features_to_dim1 = False, vertex_features_to_higher_dim = False, edge_attr_to_higher = False):

        if dataset == "ogbg-molhiv":
            dataset = "MOLHIV"
        self.dataset = dataset

        self.task_type = 'classification'

        if max_struct_dim > 0:
            self.name = f"{dataset}_{'cliques' if use_cliques else 'cycles'}-{max_struct_dim}"
            if edge_features_to_dim1:
                self.name += "_dim1features"
            if vertex_features_to_higher_dim:
                self.name += "_dim>1features"
            if edge_attr_to_higher and not use_cliques:
                self.name += "_edge_attr_to_higher"
        else:
            self.name = dataset

        self.max_struct_dim = max_struct_dim
        self.use_cliques = use_cliques
        self.raw_dataset = None
        self.edge_features_to_dim1 = edge_features_to_dim1
        self.vertex_features_to_higher_dim = vertex_features_to_higher_dim
        self.edge_attr_to_higher = edge_attr_to_higher

        super().__init__(root, None, None, None)
        print("LOADING FROM:", self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])

        print(self.data)
        if dataset in tvt_datasets:
            with open(self.processed_paths[1], "rb") as file:
                self.split_idx = pickle.load(file)

        # Get nr_classes and nr_features
        self.nr_features = self.data.x.shape[1]

        self.nr_classes = int(max(self.data.y) + 1)
        if dataset == "MOLHIV":
            self.nr_classes = 1
            self.eval_metric = "rocauc"

    @property
    def metric(self):
        if self.dataset in implemented_TU_datasets:
            return "accuracy"
        elif self.dataset == "MOLHIV":
            return "roc-auc"
        else:
            raise NotImplementedError("For this dataset no metric is defined")

    def get_raw_dataset_object(self):
        if self.raw_dataset is not None:
            return self.raw_dataset

        if self.dataset in implemented_TU_datasets:
            self.raw_dataset = torch_geometric.datasets.TUDataset(root=self.raw_dir, name=self.dataset)
        elif self.dataset == "MOLHIV":
            self.raw_dataset = PygGraphPropPredDataset(root=self.raw_dir, name = "ogbg-molhiv")
            self.split_idx = self.raw_dataset.get_idx_split()
        else:
            raise NotImplementedError("This dataset is not implemented")
        return self.raw_dataset

    @property
    def raw_file_names(self):
        print("raw_file_names")
        return [f"{self.name}"]

    # Overwrite _process so processing does not run every time (see: https://github.com/pyg-team/pytorch_geometric/issues/1802)
    def _process(self):
        makedirs(self.processed_dir)
        if len(glob.glob(os.path.join(self.processed_dir, '*.pkl'))) > 0:
            return
        self.process()

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, "processed")

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_paths(self):
        return [os.path.join(self.root, self.name, "processed", "data.pkl"),
            os.path.join(self.root, self.name, "processed", "split_idx.pkl")]


    def download(self):
        # Download to `self.raw_dir`.
        print("Download dataset")
        dataset = self.get_raw_dataset_object()

    def process(self):
        print("Starting dataset processing")
        dataset = self.get_raw_dataset_object()

        # Apply our encoding
        transformed_data_list = []
        if self.max_struct_dim > 0:
            for data in tqdm(dataset):
                if self.use_cliques:
                    transformed_data_list.append(simplex_encoding(data, self.max_struct_dim, self.edge_features_to_dim1, self.vertex_features_to_higher_dim))
                else:
                    transformed_data_list.append(ring_encoding(data, self.max_struct_dim, self.edge_features_to_dim1, self.vertex_features_to_higher_dim, self.edge_attr_to_higher))
                
        else:
            transformed_data_list = [data for data in dataset]

        data, slices = self.collate(transformed_data_list)
        print("Storing graphs to:", self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])

        # If the datasetgraphs  requires a train, valid, test split then store the indices
        if self.dataset in tvt_datasets:
            print("Storing index:", self.processed_paths[1])
            with open(self.processed_paths[1], "wb") as file:
                pickle.dump(dataset.get_idx_split() , file)
