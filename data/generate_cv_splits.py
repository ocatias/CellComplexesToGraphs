"""
Generate split index for all dataset for which we use CV.
"""

import os
import json
from numpy import int32

import torch_geometric

from data.utils import cv_split, stratified_data_split

DATA_PATH = r'./datasets'
datasets = ["MUTAG", "PROTEINS", "NCI1", "NCI109"]
folds = 10
seed = 42

tt_path = os.path.join(DATA_PATH, "Train_Test_Splits")
tvt_path = os.path.join(DATA_PATH, "Train_Val_Test_Splits")
if not os.path.isdir(tt_path):
    os.mkdir(tt_path)
if not os.path.isdir(tvt_path):
    os.mkdir(tvt_path)


for dataset_name in datasets:
    dataset = torch_geometric.datasets.TUDataset(root=DATA_PATH, name=dataset_name)
    for fold in range(folds):
        train_index, test_index = cv_split(dataset, seed, fold, folds)

        # Train and Test splits
        with open(os.path.join(tt_path, f"{dataset_name}_fold_{fold}_of_{folds}_train.json"), "w") as file:
            json.dump(list(train_index.tolist()), file)
        with open(os.path.join(tt_path, f"{dataset_name}_fold_{fold}_of_{folds}_test.json"), "w") as file:
            json.dump(list(test_index.tolist()), file)

        # Indices to data so we can split again
        training_index_y = [(idx, dataset[idx].y) for idx in train_index]

        # Train, Val and Test splits
        train_index, val_index = stratified_data_split(training_index_y, seed)

        with open(os.path.join(tvt_path, f"{dataset_name}_fold_{fold}_of_{folds}_train.json"), "w") as file:
            json.dump(list(train_index), file)
        with open(os.path.join(tvt_path, f"{dataset_name}_fold_{fold}_of_{folds}_valid.json"), "w") as file:
            json.dump(list(val_index), file)
        with open(os.path.join(tvt_path, f"{dataset_name}_fold_{fold}_of_{folds}_test.json"), "w") as file:
            json.dump(list(test_index.tolist()), file)
