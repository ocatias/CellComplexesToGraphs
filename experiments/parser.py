"""
Helper functions that do argument parsing for experiments.
"""

import argparse
import yaml
import sys
from copy import deepcopy

def parse_args_single_fold(parser):
    """
    Additional options for a single fold
    # """
    parser.add_argument('--clean', type=int, default=0,
                    help='Split of a part of the training set to use as validation set (leaving the validation fold untouched)')
    parser.add_argument('--tune', type=int, default=0,
                    help='Tune on the validation set and do not perform inference on the test set (purely used to find good parameters)')
    parser.add_argument('--repeats', type=int, default=5,
                    help='Number of time to repeat the training + test set evaluation (only relevant when --clean and --tune)')

def parse_args(passed_args=None, single_fold=False, kernel=False):
    """
    Parse command line arguments. Allows either a config file (via "--config path/to/config.yaml")
    or for all parameters to be set directly.
    A combination of these is NOT allowed.
    Partially from: https://github.com/twitter-research/cwn/blob/main/exp/parser.py
    """

    parser = argparse.ArgumentParser(description='An experiment.')

    # Config file to load
    parser.add_argument('--config', dest='config_file', type=argparse.FileType(mode='r'),
                        help='Path to a config file that should be used for this experiment. '
                        + 'CANNOT be combined with explicit arguments')

    # Parameters to be set directly
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed to set (default: 42)')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                            help='dataset name (default: MUTAG)')
    parser.add_argument('--fold', type=int, default=0,
                            help='fold index for k-fold cross-validation experiments')
    parser.add_argument('--folds', type=int, default=10,
                            help='The number of folds to run on in cross validation experiments')

    if not kernel:
        parser.add_argument('--lr', type=float, default=0.001,
                            help='learning rate (default: 0.001)')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='input batch size for training (default: 32)')
        parser.add_argument('--epochs', type=int, default=100,
                            help='number of epochs to train (default: 100)')
        
        parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
        parser.add_argument('--model', type=str, default='GCN',
                        help='model, possible choices: default')
        parser.add_argument('--task_type', type=str, default='classification',
                        help='task_type, possible choices: classification')
        parser.add_argument('--lr_scheduler', type=str, default='StepLR',
                        help='learning rate decay scheduler (default: StepLR)')
        parser.add_argument('--lr_scheduler_decay_steps', type=int, default=50,
                            help='number of epochs between lr decay (default: 50)')
        parser.add_argument('--lr_scheduler_decay_rate', type=float, default=0.5,
                            help='strength of lr decay (default: 0.5)')
        parser.add_argument('--lr_scheduler_min', type=float, default=0.00001,
                            help='min LR for `ReduceLROnPlateau` lr decay (default: 1e-5)')
        
        parser.add_argument('--tracking', type=int, default=1,
                            help='If 0 runs without tracking')
        parser.add_argument('--drop_out', type=float, default=0.0,
                            help='dropout rate (default: 0.0)')
        parser.add_argument('--emb_dim', type=int, default=64,
                            help='dimensionality of hidden units in models (default: 64)')
        parser.add_argument('--num_layers', type=int, default=5,
                            help='number of message passing layers (default: 5)')
    else:
        parser.add_argument('--base_kernel', type=str, default='WL-VH',
                            help='')
        parser.add_argument('--wl_iter', type=int, default=1,
                            help='')


    parser.add_argument('--cliques', type=int, default=0,
                        help='Attach cells to all cliques of size up to k.')
    parser.add_argument('--rings', type=int, default=0,
                        help='Attach cells to all rings of size up to k.')

    parser.add_argument('--aggr_edge_atr', type=int, default=0,
                        help='')
    parser.add_argument('--aggr_vertex_feat', type=int, default=0,
                        help='')
    parser.add_argument('--explicit_pattern_enc', type=int, default=0,
                        help='')
    parser.add_argument('--edge_attr_in_vertices', type=int, default=0,
                        help='')
    parser.add_argument('--max_struct_size', type=int, default=0,
                        help='Maximum size of the structure to attach cells to. If it is non-zero then cycle encoding will be used, except if --cliques 1 then clique encoding will be used')

    if single_fold:
        parse_args_single_fold(parser)

    original_args = sys.argv

    # Load partial args instead of command line args (if they are given)
    if passed_args is not None:
        original_args = passed_args
        # Transform dict to list of args
        list_args = []
        for key,value in passed_args.items():
            # The case with "" happens if we want to pass an argument that has no parameter
            list_args += [key, str(value)]

        args = parser.parse_args(list_args)
    else:
        args = parser.parse_args()
        
    original_args = deepcopy(args)
    print(original_args)


    args.__dict__["use_rings"] = args.rings == 1
    args.__dict__["use_cliques"] = args.cliques == 1
    assert not (args.__dict__["use_rings"] and args.__dict__["use_cliques"])

    args.__dict__["use_aggr_edge_atr"] = args.aggr_edge_atr == 1
    args.__dict__["use_aggr_vertex_feat"] = args.aggr_vertex_feat == 1
    args.__dict__["use_explicit_pattern_enc"] = args.explicit_pattern_enc == 1
    args.__dict__["use_edge_attr_in_vertices"] = args.edge_attr_in_vertices == 1

    if not kernel:
        args.__dict__["use_tracking"] = args.tracking == 1


    if single_fold:
        args.__dict__["clean"] = args.clean == 1
        args.__dict__["tune"] = args.tune == 1


    # https://codereview.stackexchange.com/a/79015
    # If a config file is provided, write it's values into the arguments
    if args.config_file:
        data = yaml.load(args.config_file)
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        for key, value in data.items():
                arg_dict[key] = value

    return original_args, args
