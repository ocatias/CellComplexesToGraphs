"""
Evaluate a model with "clean" cross validation.
For a given fold splits the data into training and testing set.
Then splits a validation set from the training set and performs model selection on the validation set.
Finally, for the best parameters trains a model multiple times and evaluates it on the test set
"""

import argparse
import os
import glob
from sklearn import datasets
import yaml
import csv

import numpy as np
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed

from experiments.run_single_fold import main as run_gnn_experiment
from experiments.run_graphkernel_single_fold import main as run_kernel_experiment

folds = 10
RESULTS_PATH = r"./results"


def run(fold, grid, args):
    prev_params = []

    # FIND PARAMETERS
    for c in range(args.candidates):
        # Generate a new parameter

        # Set seed randomly, because model training sets seeds and we want different parameters every time
        np.random.seed()

        param = np.random.choice(grid, 1)
        while param in prev_params:
            param = np.random.choice(grid, 1)

        prev_params.append(param)
        print(f"Fold {fold}, candidate {c} selected: {param}")
        # Evaluate this parameter

        param_dict = {
            "--fold": fold,
            "--folds": folds,
            "--dataset": args.dataset,
            "--clean": 1,
            "--tune": 1}
        for key, value in param[0].items():
            param_dict["--" + str(key)] = str(value)

        if args.kernel:
            run_kernel_experiment(param_dict)
        else:
            param_dict["--tracking"] = 0
            run_gnn_experiment(param_dict)

        if len(prev_params) >= len(grid):
            break

    # EVALTE BEST PARAMETRS
    param_dict_test = {
            "--fold": fold,
            "--folds": folds,
            "--dataset": args.dataset,
            "--clean": 1,
            "--tune": 0,
            "--repeats": args.repeats}
    if args.kernel:
        # Need to give the script the correct base_kernel, so it knows where to look for the params
        param_dict_test["--base_kernel"] = grid[0]["base_kernel"]
        run_kernel_experiment(param_dict_test)
    else:
        param_dict["--tracking"] = 0
        run_gnn_experiment(param_dict_test)

    return None

def main():
    parser = argparse.ArgumentParser(description='An experiment.')
    parser.add_argument('-grid', dest='grid_file', type=argparse.FileType(mode='r'),
                    help="Path to a .yaml file that contains the parameters grid.")
    parser.add_argument('-dataset', type=str)
    parser.add_argument('--workers', type=int, default=4,
                    help="Number of models to train in parallel.")
    parser.add_argument('--candidates', type=int, default=20,
                    help="Number of parameter combinations to try per fold.")
    parser.add_argument('--repeats', type=int, default=5,
                    help="Number of times to repeat the final model training and evaluation per fold.")
    parser.add_argument('--kernel', action='store_true')
    args = parser.parse_args()
    grid_raw = yaml.load(args.grid_file)
    grid = ParameterGrid(grid_raw)

    Parallel(n_jobs=args.workers, prefer="threads")(delayed(run)(fold, grid, args) for fold in range(folds))

    # Collect results of all runs
    accuracies = []
    for fold in range(folds):
        if not args.kernel:
            file_path = os.path.join(RESULTS_PATH, args.dataset, f"Fold_{fold}_results.csv")
        else:
            file_path = os.path.join(RESULTS_PATH, grid_raw["base_kernel"][0], args.dataset, f"Fold_{fold}_results.csv")

        with open(file_path) as file:
            for line in file.readlines():
                accuracies.append(float(line))

    avg = np.average(accuracies)
    std = np.std(accuracies)

    print("\n\n\n\nFINAL RESULT")
    print(f"avg: {avg}\nstd: {std}")

    if not args.kernel:
        file_path = os.path.join(RESULTS_PATH, args.dataset, "final_results.txt")
    else:
        file_path = os.path.join(RESULTS_PATH, grid_raw["base_kernel"][0], args.dataset, "final_results.txt")

    with open(file_path, "w") as file:
        file.write(f"avg: {avg}\nstd: {std}")

if __name__ == "__main__":
    main()
    
