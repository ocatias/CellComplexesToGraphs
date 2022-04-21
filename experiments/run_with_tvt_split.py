"""
Train and evaluate a GNN with a train, validation and test split
"""
import os

from torch_geometric.loader import DataLoader
import wandb
import numpy as np

from experiments.parser import parse_args
from experiments.train_utils import train, eval, set_seeds, get_model_optim_scheduler, get_device
from data.utils import separate_data
from data.dataset import CellEncodedDataset, tvt_datasets

DATA_PATH = r'./datasets'
MODELS_PATH = r'./models'

def get_data_loaders(args):
    dataset = CellEncodedDataset(DATA_PATH, args.dataset, args.max_struct_size, args.use_cliques, args.edge_features_to_dim1, args.vertex_features_to_higher_dim)

    split_idx = dataset.split_idx
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size)

    return dataset, train_loader, valid_loader,test_loader

def main(args):
    if args.dataset not in tvt_datasets:
        raise ValueError("run_with_tvt_split is only supposed to run on datasets that have a train, valid, test split.")

    set_seeds(args.seed)
    device = get_device(args)
    dataset, train_loader, valid_loader,test_loader = get_data_loaders(args)
    model, optimizer, scheduler = get_model_optim_scheduler(dataset, args, device)

    print("Dataset:", dataset.name)
    print(f"Nr classes {dataset.nr_classes}\nNr features: {dataset.nr_features}")

    # Initialize WandB tracking
    if args.use_tracking:
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(
            config = args,
            project = "GT-RL22 TVT")

    train_curve, valid_curve, test_curve = [], [], []
    for epoch in range(1, args.epochs + 1):
        train_dict = train(model, device, train_loader, optimizer, dataset.metric, args.use_tracking)
        val_dict = eval(model, device, valid_loader, dataset.metric)
        test_dict = eval(model, device, test_loader, dataset.metric)

        print(f"Train {dataset.metric}: {train_dict[dataset.metric]:.4f}\tVal {dataset.metric}: {val_dict[dataset.metric]:.4f}")
        train_curve.append(train_dict[dataset.metric])
        valid_curve.append(val_dict[dataset.metric])
        test_curve.append(test_dict[dataset.metric])

        if args.use_tracking:
            wandb.log({
                "Epoch": epoch,
                "Train/Loss": train_dict["total_loss"],
                f"Train/{dataset.metric}": train_dict[dataset.metric],
                "Val/Loss": val_dict["total_loss"],
                f"Val/{dataset.metric}": val_dict[dataset.metric]})

        if scheduler is not None:
            scheduler.step()

    best_val_epoch = np.argmax(valid_curve)
    result_train = train_curve[best_val_epoch]
    result_best_val = valid_curve[best_val_epoch]
    result_test = test_curve[best_val_epoch]

    if args.use_tracking:
        wandb.log({f"Final/Train_{dataset.metric}": result_train,
            f"Final/Valid_{dataset.metric}": result_best_val,
            f"Final/Test_{dataset.metric}": result_test})

    print("\n\n\n\n")
    print("RESULTS:")
    print("_____________________________________________")
    print(f"SETTINGS: {args}\n")
    print("Last epoch:")
    print(f"Train {dataset.metric}: {train_curve[-1]:.4f}")
    print(f"Valid {dataset.metric}: {valid_curve[-1]:.4f}")
    print(f"Test {dataset.metric}: {test_curve[-1]:.4f}")
    print("\nSelected by best validation value:")
    print(f"Train {dataset.metric}: {result_train:.4f}")
    print(f"Valid {dataset.metric}: {result_best_val:.4f}")
    print(f"Test {dataset.metric}: {result_test:.4f}")
    print("_____________________________________________")

if __name__ == "__main__":
    print("STARTING")
    original_args, args = parse_args()
    main(args)
