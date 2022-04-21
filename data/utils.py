import json
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

DATA_PATH = r'./datasets'

def edge_tensor_to_list(edge_tensor):
    edge_list = []
    for i in range(edge_tensor.shape[1]):
        edge_list.append((int(edge_tensor[0, i]), int(edge_tensor[1, i])))

    return edge_list

def separate_data(graph_list, seed, fold_idx, nr_folds = 10):
    """
    Split dataset into folds
    Adapted from: https://github.com/weihua916/powerful-gnns/blob/master/util.py
    """
    assert 0 <= fold_idx and fold_idx < nr_folds, "fold_idx must be from 0 to nr_folds."
    skf = StratifiedKFold(n_splits=nr_folds, shuffle = True, random_state = seed)

    idx_list = []
    y = np.concatenate([d.y for d in graph_list]) 

    for idx in skf.split(np.zeros(len(graph_list)), y):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list

def cv_split(graph_list, seed, fold_idx, nr_folds = 10):
    """
    Split dataset into folds
    Adapted from: https://github.com/weihua916/powerful-gnns/blob/master/util.py
    """
    assert 0 <= fold_idx and fold_idx < nr_folds, "fold_idx must be from 0 to nr_folds."
    skf = StratifiedKFold(n_splits=nr_folds, shuffle = True, random_state = seed)

    idx_list = []
    y = np.concatenate([d.y for d in graph_list]) 

    for idx in skf.split(np.zeros(len(graph_list)), y):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    return train_idx, test_idx

def stratified_data_split(training_index_y, seed):
    """
    training_index_y must have the shape of (idx, y) where idx is the position in the original dataset
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)

    indices = [d[0] for d in training_index_y]
    y = np.concatenate([d[1] for d in training_index_y]) 
    train_idx_idx, test_idx_idx = list(sss.split(indices, y))[0]
    train_index = [int(indices[idx]) for idx in train_idx_idx]
    test_index = [int(indices[idx]) for idx in test_idx_idx]
    return train_index, test_index

def load_tt_indices(dataset_name, fold, folds):
    train_path = os.path.join(DATA_PATH, "Train_Test_Splits", f"{dataset_name}_fold_{fold}_of_{folds}_train.json")
    test_path = os.path.join(DATA_PATH, "Train_Test_Splits", f"{dataset_name}_fold_{fold}_of_{folds}_test.json")

    with open(train_path) as file:
        train_idx = json.load(file)
    with open(test_path) as file:
        test_idx = json.load(file)

    return train_idx, test_idx

def load_tvt_indices(dataset_name, fold, folds):
    train_path = os.path.join(DATA_PATH, "Train_Val_Test_Splits", f"{dataset_name}_fold_{fold}_of_{folds}_train.json")
    valid_path = os.path.join(DATA_PATH, "Train_Val_Test_Splits", f"{dataset_name}_fold_{fold}_of_{folds}_valid.json")
    test_path = os.path.join(DATA_PATH, "Train_Val_Test_Splits", f"{dataset_name}_fold_{fold}_of_{folds}_test.json")

    with open(train_path) as file:
        train_idx = json.load(file)
    with open(valid_path) as file:
        valid_idx = json.load(file)
    with open(test_path) as file:
        test_idx = json.load(file)

    return train_idx, valid_idx, test_idx