"""
Runs a graphkernel on a single fold of a given dataset
"""
import json
import os
import glob

import numpy as np

from grakel.datasets import fetch_dataset
from grakel.graph import Graph
from grakel.kernels import WeisfeilerLehman, VertexHistogram, ShortestPath
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from experiments.parser import parse_args
from experiments.train_utils import get_dataset
from data.utils import load_tt_indices, load_tvt_indices

DATA_PATH = r'./datasets'
RESULTS_PATH = r"./results"

def pyg_to_grakel_graph(data):
    y = int(data.y[0])

    edges = []
    for i in range(data.edge_index.shape[1]):
        edges.append((int(data.edge_index[0,i]), int(data.edge_index[1,i])))

    node_labels = {}
    for i in range(data.x.shape[0]):
        node_labels[i] = tuple(data.x[i].tolist())

    # G = Graph(edges, node_labels=node_labels)
    G = [edges, node_labels]
    
    return G, y

def transform_list_of_graphs(graph_list):
    G = []
    y = []
    for graph in graph_list:
        G_new, y_new = pyg_to_grakel_graph(graph)
        G.append(G_new)
        y.append(y_new)

    return G, np.asarray(y)

def main(args=None):
    original_args, args = parse_args(args, kernel=True, single_fold=True)
    if not args.tune:
        # Find params with best val acc
        best_acc = 0
        best_params =  None
        for param_file in glob.glob(os.path.join(RESULTS_PATH, args.base_kernel, args.dataset, f"Fold_{args.fold}", "*.json")):
            with open(param_file) as file:
                params = json.load(file)
                if params["train_acc"] > best_acc:
                    best_acc = params["train_acc"]
                    best_params = params["args"]

        print(f"Selected params with {best_acc} validation accuracy")
        print(best_params)

        # Clean the selected params
        del best_params['config_file']
        best_params["tune"] = 0

        # Ensure to use the params given by run_clean_cv
        best_params["fold"] = args.fold
        best_params["dataset"] = args.dataset
        best_params["folds"] = args.folds

        cleaned_params = {}
        for key, value in best_params.items():
            cleaned_params["--" + key] = value

        original_args, args = parse_args(cleaned_params, kernel=True, single_fold=True)
        with open(os.path.join(RESULTS_PATH, args.base_kernel, args.dataset, f"Fold_{args.fold}_params.json"), "w") as file:
            json.dump(args.__dict__, file)    

    dataset = get_dataset(DATA_PATH, args)
    train_idx, val_idx, test_idx = load_tvt_indices(args.dataset, args.fold, args.folds)
    train_idx = train_idx + val_idx

    train_data, test_data = dataset[train_idx], dataset[test_idx]

    assert len(dataset) == len(train_idx) + len(test_idx)

    G_train, y_train = transform_list_of_graphs(train_data)
    G_test, y_test = transform_list_of_graphs(test_data)
    
    print("Fit transform")
    if args.base_kernel == "WL-VH": 
        print("WL + VertexHistogram")
        kernel = WeisfeilerLehman(n_iter=args.wl_iter, normalize=True, base_graph_kernel=VertexHistogram)
    else:
        print("WL + Shortest Path")
        kernel = WeisfeilerLehman(n_iter=args.wl_iter, normalize=True, base_graph_kernel=ShortestPath)
    print("WL Iterations: ", args.wl_iter)

    K_train = kernel.fit_transform(G_train)
    K_test = kernel.transform(G_test)

    print("Running SVC")
    clf = SVC(kernel='precomputed')

    clf.fit(K_train, y_train)
    SVC(kernel='precomputed')

    y_pred_train = clf.predict(K_train)
    y_pred_test = clf.predict(K_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print("Training: %2.2f %%" %(round(train_acc*100)))
    print("Testing: %2.2f %%" %(round(test_acc*100)))


    path_results = os.path.join(RESULTS_PATH, args.base_kernel)
    if not os.path.isdir(path_results):
        os.mkdir(path_results)
    path_results = os.path.join(path_results, args.dataset)
    if not os.path.isdir(path_results):
            os.mkdir(path_results)

    if args.tune:   
        directory = os.path.join(path_results, f"Fold_{args.fold}")
        if not os.path.isdir(directory):
            os.mkdir(directory)

        nr_files = len(glob.glob(os.path.join(directory, "*.json")))
        name = f"result_{nr_files}.json"
        data = {"args": original_args.__dict__,  "train_acc": train_acc}
        print(data)
        with open(os.path.join(directory, name), "w") as file:
            json.dump(data, file)
    else:
        with open(os.path.join(path_results, f"Fold_{args.fold}_results.csv"), "a") as file:
            file.write(str(test_acc) + "\n")


if __name__ == "__main__":
    main()
