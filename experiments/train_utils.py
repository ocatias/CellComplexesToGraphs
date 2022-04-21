"""
Functions to help during training such as training for one epoch, or evaluating

Partially from: https://github.com/twitter-research/cwn/blob/main/exp/train_utils.py
"""
import random
import os

import numpy as np
import torch
from tqdm import tqdm
import wandb
import torch.optim as optim
from ogb.graphproppred import Evaluator
import torch_geometric
from ogb.graphproppred import PygGraphPropPredDataset

from models.gcn import GCN
from models.gin import GIN, GINMOLHIV
from SPE.clique_encoding import CellularCliqueEncoding
from SPE.ring_encoding import CellularRingEncoding

implemented_TU_datasets = ["MUTAG", "PROTEINS", "IMDB-BINARY", "COLLAB", "NCI1", "NCI109"]
# Datasets that are supposed to use a train, validation and test split
tvt_datasets = ["ogbg-molhiv"]

cls_criterion = torch.nn.CrossEntropyLoss()
bicls_criterion = torch.nn.BCEWithLogitsLoss()

def get_loss_fct(metric):
    if metric == 'accuracy':
        return cls_criterion
    elif metric == 'roc-auc':
        return bicls_criterion
    else:
        raise ValueError("Selected an invalid metric type")

def compute_loss_predictions(batch, model, metric, device, loss_fn, tracking_dict):
    batch_size = batch.y.shape[0]
    batch = batch.to(device)
    predictions = model(batch)

    if metric == 'accuracy':
        loss = loss_fn(predictions, batch.y.view(-1,))
    else:
        loss = loss_fn(predictions, batch.y.float())

    if metric == 'accuracy':
        tracking_dict["correct_classifications"] += torch.sum(predictions.argmax(dim=1)== batch.y).item()
    elif metric == 'roc-auc':
        tracking_dict["y_preds"] += predictions.cpu()
        tracking_dict["y_true"] += batch.y.cpu()

    tracking_dict["batch_losses"].append(loss.item())
    tracking_dict["total_loss"] += loss.item()*batch_size

    return loss

def get_tracking_dict():
    return {"correct_classifications": 0, "y_preds":[], "y_true":[],  "total_loss":0, "batch_losses":[]}


def compute_final_tracking_dict(tracking_dict, output_dict, loader, metric):
    output_dict["total_loss"] = tracking_dict["total_loss"] / len(loader.dataset)
    if metric == 'accuracy':
        output_dict["accuracy"] = tracking_dict["correct_classifications"] / len(loader.dataset)
    elif metric == 'roc-auc':
        evaluator = Evaluator(name = "ogbg-molhiv")
        y_preds = torch.concat(tracking_dict["y_preds"])
        y_true = torch.concat(tracking_dict["y_true"])
        y_preds = torch.unsqueeze(y_preds, dim = 1)
        y_true = torch.unsqueeze(y_true, dim = 1)
        result_dict = evaluator.eval({"y_true": y_true, "y_pred": y_preds})
        output_dict["roc-auc"] = result_dict["rocauc"]
    return output_dict

def train(model, device, loader, optimizer, metric='accuracy', use_tracking=False):
    """
        Performs one training epoch, i.e. one optimization pass over the batches of a data loader.
    """

    loss_fn = get_loss_fct(metric)
    model.train()

    tracking_dict = get_tracking_dict()
    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        loss = compute_loss_predictions(batch, model, metric, device, loss_fn, tracking_dict)

        loss.backward()
        optimizer.step()

        if use_tracking:
            wandb.log({'Train/BatchLoss': loss.item()})

    trainig_data = {"batch_loss":tracking_dict["batch_losses"]}
    compute_final_tracking_dict(tracking_dict, trainig_data, loader, metric)
    return trainig_data


def eval(model, device, loader, metric):
    """
        Evaluates a model over all the batches of a data loader.
    """
    loss_fn = get_loss_fct(metric)
    model.eval()

    tracking_dict = get_tracking_dict()
    for step, batch in enumerate(loader):
        with torch.no_grad():
            loss = compute_loss_predictions(batch, model, metric, device, loss_fn, tracking_dict)

    eval_dict = compute_final_tracking_dict(tracking_dict, {}, loader, metric)
    return eval_dict

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_model(args, nr_classes, nr_features):
    # For cliques max cell dim is equal to max structure size
    # For rings the max structure size is always 2
    max_struct_size = args.max_struct_size
    if max_struct_size > 0 and not args.cliques:
        max_struct_size = 2

    if args.model == "GIN":
        return GIN(nr_features, args.num_layers, args.emb_dim, nr_classes, dropout_rate=args.drop_out, max_cell_dim=max_struct_size, dimensional_pooling=args.use_explicit_pattern_enc)
    # elif args.model == "GINMOLHIV":
    #     return GIN(nr_features, args.num_layers, args.emb_dim, nr_classes, dropout_rate=args.drop_out, max_cell_dim=max_struct_size)
    elif args.model == "GCN":
        return GCN(nr_features, nr_classes)

def get_device(args):
    device = torch.device(
        "cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print("Device: {0}".format(device))
    return device

def get_model_optim_scheduler(dataset, args, device):
    model = get_model(args, dataset.num_classes, dataset.num_node_features)
    model.to(device)

    # OPTIMIZER
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # LEARNING RATE SCHEDULER
    # instantiate learning rate decay
    if args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_scheduler_decay_steps,
                                                    gamma=args.lr_scheduler_decay_rate)
    elif args.lr_scheduler == 'None':
        scheduler = None
    else:
        raise NotImplementedError(f'Scheduler {args.lr_scheduler} is not currently supported.')
    return model, optimizer, scheduler

def get_dataset(DATA_PATH, args):
    encoder = None
    if args.use_cliques:
        encoder = CellularCliqueEncoding(args.max_struct_size, aggr_edge_atr=args.use_aggr_edge_atr, aggr_vertex_feat=args.use_aggr_vertex_feat,
            explicit_pattern_enc=args.use_explicit_pattern_enc, edge_attr_in_vertices=args.use_edge_attr_in_vertices)
    elif args.use_rings:
        encoder = CellularRingEncoding(args.max_struct_size, aggr_edge_atr=args.use_aggr_edge_atr, aggr_vertex_feat=args.use_aggr_vertex_feat,
            explicit_pattern_enc=args.use_explicit_pattern_enc, edge_attr_in_vertices=args.use_edge_attr_in_vertices)

    if encoder is None:
        dir = os.path.join(DATA_PATH, args.dataset)
    else:
        dir = os.path.join(DATA_PATH, args.dataset + "_" + repr(encoder))

    if args.dataset in implemented_TU_datasets:
        return torch_geometric.datasets.TUDataset(root=dir, name=args.dataset, pre_transform=encoder, use_node_attr=True, use_edge_attr=False)
    elif args.dataset in tvt_datasets:
        return PygGraphPropPredDataset(root=dir, name =args.dataset, pre_transform=encoder)