"""
Performs the experiment specified via parameters or config file on all folds of a dataset.
"""
import os
import copy

import numpy as np
import wandb

from experiments.parser import parse_args
from experiments.run_single_fold import run as run_experiment

def main(original_args, args):

    if args.use_tracking:
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(
            config = args,
            project = "GT-RL22 Cross-Validation")


    # The fold parameter makes no sense here, so we set it to a constant value
    args.fold = None

    # Add 
    args.__dict__["clean"] = False
    args.__dict__["tune"] = False

    args_for_single_fold_exp = copy.deepcopy(args)
    # Disable tracking on single fold runs to not confuse wandb
    args_for_single_fold_exp.use_tracking = False


    train_curves, val_curves = [], []
    for fold in range(args.folds):
        args_for_single_fold_exp.fold = fold
        print("ARGS:", args_for_single_fold_exp)
        train_results, val_results, _ = run_experiment(original_args, args_for_single_fold_exp)

        train_curves.append(train_results)
        val_curves.append(val_results)

    # Create a list of shape [folds, epochs]
    val_accs_per_epoch = [[val_curves[fold]["accuracy"][epoch] for fold in range(args.folds)] for epoch in range(args.epochs)]
    avg_val_acc = [np.mean(val_accs) for val_accs in val_accs_per_epoch]
    best_val_acc_epoch = np.argmax(avg_val_acc)

    # Get results i.e. the average accuracy of the best epoch and the std of that epoch
    accuracy = avg_val_acc[best_val_acc_epoch]
    std = np.std(val_accs_per_epoch[best_val_acc_epoch])

    # TRACKING
    if args.use_tracking:
        for epoch in range(args.epochs):
            for fold in range(args.folds):
                wandb.log({
                    f"Train/Loss_{fold}": train_curves[fold]["total_loss"][epoch],
                    f"Train/Accuracy_{fold}": train_curves[fold]["accuracy"][epoch],
                    f"Val/Loss_{fold}": val_curves[fold]["total_loss"][epoch],
                    f"Val/Accuracy_{fold}": val_curves[fold]["accuracy"][epoch]}, step=epoch)

            wandb.log({"Final/Average_Accuracy": avg_val_acc[epoch]}, step=epoch)

        wandb.log({"Final/Best_avg_acc": accuracy,
            "Final/Best_avg_acc_std": std})

        wandb.run.summary["Best_Accuracy"] = accuracy
        wandb.run.summary["STD"] = std

    print("\n\n\n\n")
    print("RESULTS:")
    print("_____________________________________________")
    print(f"SETTINGS: {args}\n")

    print(f"ACCURACY: {accuracy:.4f}±{std:.4f}")
    print("_____________________________________________")

def run_with_partial_args(partial_args):
    print("partial_args", partial_args)
    args = parse_args(partial_args)
    main(args)
    wandb.finish()

if __name__ == "__main__":
    original_args, args = parse_args()
    main(original_args, args)
    wandb.finish()
