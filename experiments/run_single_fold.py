"""
Script to train and evaluate the performance of GNNs on different datasets.
Trains on the specified fold of the dataset (see --fold parameter)

Parameters can be given as command line arguments or by providing a config file, see -h

Partially based on: https://github.com/twitter-research/cwn/blob/main/exp/run_exp.py
"""

import os
import glob
import json

import numpy as np
import torch
import torch_geometric
from torch_geometric.loader import DataLoader
import wandb

from experiments.parser import parse_args
from experiments.train_utils import train, eval, set_seeds, get_model_optim_scheduler, get_device, get_dataset, tvt_datasets
from misc.utils import list_of_dictionary_to_dictionary_of_lists
from data.utils import load_tt_indices, load_tvt_indices

DATA_PATH = r'./datasets'
RESULTS_PATH = r"./results"

def get_data_loaders(args):
    dataset = get_dataset(DATA_PATH, args)
    test_loader = None

    if args.clean:
        train_idx, val_idx, test_idx = load_tvt_indices(args.dataset, args.fold, args.folds)

        train_loader = DataLoader(dataset[train_idx], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(dataset[val_idx], batch_size=args.batch_size)

        if not args.tune:
            test_loader =  DataLoader(dataset[test_idx], batch_size=args.batch_size)
    else:
        train_idx, val_idx = load_tt_indices(args.dataset, args.fold, args.folds)
        train_loader = DataLoader(dataset[train_idx], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(dataset[val_idx], batch_size=args.batch_size)

    return dataset, train_loader, val_loader, test_loader

def run(original_args, args):
    if args.dataset  in tvt_datasets:
        raise ValueError("run_single_fold is only supposed to run on datasets that do not have a train, valid, test split.")

    set_seeds(args.seed)
    device = get_device(args)
    dataset, train_loader, val_loader, test_loader = get_data_loaders(args)
    model, optimizer, scheduler = get_model_optim_scheduler(dataset, args, device)

    print("Dataset:", dataset.name)
    print(f"Nr classes {dataset.num_classes}\nNr features: {dataset.num_node_features}")
    print(args)
    # Initialize WandB tracking
    if args.use_tracking:
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(
            config = args,
            project = "GT-RL22 Individual Folds")

    training_results, val_results, test_results = [], [], []
    for epoch in range(1, args.epochs + 1):
        train_dict = train(model, device, train_loader, optimizer, "accuracy", args.use_tracking)
        val_dict = eval(model, device, val_loader, "accuracy")
        if test_loader is not None:
            test_dict = eval(model, device, test_loader, "accuracy")
            test_results.append(test_dict)
            print(f"[{epoch}/{args.epochs}]Train acc: {train_dict['accuracy']:.4f}\tVal acc: {val_dict['accuracy']:.4f}\tTest acc: {test_dict['accuracy']}")

        else:
            print(f"[{epoch}/{args.epochs}]Train acc: {train_dict['accuracy']:.4f}\tVal acc: {val_dict['accuracy']:.4f}")

        training_results.append(train_dict)
        val_results.append(val_dict)

        if args.use_tracking:
            wandb.log({
                "Epoch": epoch,
                "Train/Loss": train_dict["total_loss"],
                "Train/Accuracy": train_dict["accuracy"],
                "Val/Loss": val_dict["total_loss"],
                "Val/Accuracy": val_dict["accuracy"]})

        if scheduler is not None:
            scheduler.step()

    training_results = list_of_dictionary_to_dictionary_of_lists(training_results)
    val_results = list_of_dictionary_to_dictionary_of_lists(val_results)
    best_val_epoch = np.argmax(val_results["accuracy"])
    val_results["best_epoch"] = best_val_epoch

    print("\n\nBest validation epoch:")
    print(f"Train acc: {training_results['accuracy'][best_val_epoch]:.4f}"
        + f"\nVal acc: {val_results['accuracy'][best_val_epoch]:.4f}")

    if test_loader is not None:
        test_results = list_of_dictionary_to_dictionary_of_lists(test_results)
        print(f"Test acc: {test_results['accuracy'][best_val_epoch]:.4f}")

    if args.tune:
        store_results(original_args.__dict__, args, val_results['accuracy'][best_val_epoch])

    if args.use_tracking:
        wandb.finish()

    return training_results, val_results, test_results


def store_results(args_to_store, args, val_acc):
    """
    Store the parameters and validation accuracy
    """
    dir = os.path.join(RESULTS_PATH, args.dataset, f"Fold_{args.fold}")
    if not os.path.isdir(dir):
        os.makedirs(dir)

    nr_files = len(glob.glob(os.path.join(dir, "*.json")))
    name = f"result_{nr_files}.json"
    data = {"args": args_to_store,  "val_acc": val_acc}
    print(data)
    with open(os.path.join(dir, name), "w") as file:
        json.dump(data, file)

def main(args=None):
    original_args, args = parse_args(args, single_fold=True)

    if args.clean and not args.tune:
        repeats = args.repeats

        # Find params with best val acc
        best_acc = 0
        best_params =  None
        for param_file in glob.glob(os.path.join(RESULTS_PATH, args.dataset, f"Fold_{args.fold}", "*.json")):
            with open(param_file) as file:
                params = json.load(file)
                if params["val_acc"] > best_acc:
                    best_acc = params["val_acc"]
                    best_params = params["args"]

        print(f"Selected params with {best_acc} validation accuracy")
        print(best_params)

        # Clean the selected params
        del best_params['config_file']
        best_params["tune"] = 0
        best_params["clean"] = 1
        best_params["tracking"] = 0

        # Ensure to use the params given by run_clean_cv
        best_params["fold"] = args.fold
        best_params["dataset"] = args.dataset
        best_params["folds"] = args.folds

        cleaned_params = {}
        for key, value in best_params.items():
            cleaned_params["--" + key] = value

        # Load those params 
        original_args, args = parse_args(cleaned_params, single_fold=True)
        with open(os.path.join(RESULTS_PATH, args.dataset, f"Fold_{args.fold}_params.json"), "w") as file:
            json.dump(args.__dict__, file)

        # Run repeated experiments
        for _ in range(repeats):
            training_results, val_results, test_results = run(original_args, args)
            
            best_val_epoch = val_results['best_epoch']

            print(f"val_results: {val_results}")
            print(f"test_results: {test_results}")
            print(f"best_val_epoch: {best_val_epoch}")

            test_acc = test_results["accuracy"][best_val_epoch]
            with open(os.path.join(RESULTS_PATH, args.dataset, f"Fold_{args.fold}_results.csv"), "a") as file:
                file.write(str(test_acc) + "\n")
        
    else:
        run(original_args, args)

if __name__ == "__main__":
    main()
