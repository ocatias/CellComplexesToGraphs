"""
Evaluate a model with "clean" cross validation.
Based on experiments/run_clean_cv.py
"""

import argparse
import os
import yaml
import glob
import json

from joblib import Parallel, delayed
import numpy as np
from sklearn.model_selection import ParameterGrid

from run_exp import main as run_experiment
from parser import get_parser, validate_args

folds = 10
RESULTS_PATH = "../../results/CWN/"


def run(fold, grid, args, params_to_tune):
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

        params = [
            "--fold", str(fold),
            "--folds", str(folds),
            "--dataset", args.dataset,
            "--tune_params", "1"]
        for key, value in param[0].items():
            params += ["--" + str(key), str(value)]

        parser = get_parser()
        print("param_dict", params)
        exp_args = parser.parse_args(params)

        run_experiment(exp_args)

        if len(prev_params) >= len(grid):
            break

    # EVALTE BEST PARAMETRS
    for _ in range(args.repeats):

        # Select params with highest val score
        best_acc = 0
        best_params =  None

        for param_file in glob.glob(os.path.join(RESULTS_PATH, args.dataset, f"Fold_{fold}", "*.json")):
            with open(param_file) as file:
                params = json.load(file)
                if params["vaL_result"] > best_acc:
                    best_acc = params["vaL_result"]
                    best_params = params["args"]

        best_params["tune_params"] = "1"

        params_test = [
            "--fold", str(fold),
            "--folds", str(folds),
            "--dataset", args.dataset,
            "--tune_params", "0"]
        for key, value in best_params.items():
            if value == None:
                continue

            if key in params_to_tune:
                params_test += ["--" + key, str(value)]
            else: 
                # NEED TO ENSURE THAT WE DO NOT TUNE "readout_dims" OTHERWISE THIS WILL IGNORE THE PARAM
                # print(key, value)
                # params_test += ["--" + key, (value)]
                pass

        print("\n\n\n\nPRE PARSING:\n", params_test)
        parser = get_parser()
        exp_test_args = parser.parse_args(params_test)
        print("\n\nexp_test_args:\n", exp_test_args)

        run_experiment(exp_test_args)
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
    args = parser.parse_args()
    grid_raw = yaml.load(args.grid_file)
    grid = ParameterGrid(grid_raw)

    params = grid_raw.keys()

    Parallel(n_jobs=args.workers, prefer="threads")(delayed(run)(fold, grid, args, params) for fold in range(folds))

    # Collect results of all runs
    accuracies = []
    for fold in range(folds):
        with open(os.path.join(RESULTS_PATH, args.dataset, f"Fold_{fold}_results.csv")) as file:
            for line in file.readlines():
                accuracies.append(float(line))

    avg = np.average(accuracies)
    std = np.std(accuracies)

    print("\n\n\n\nFINAL RESULT")
    print(f"avg: {avg}\nstd: {std}")
    with open(os.path.join(RESULTS_PATH, args.dataset, "final_results.txt"), "w") as file:
        file.write(f"avg: {avg}\nstd: {std}")

if __name__ == "__main__":
    main()
    