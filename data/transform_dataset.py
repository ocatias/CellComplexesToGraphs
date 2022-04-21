"""
Apply Cell Encoding to a graph and store the graph so it can be loaded for future experiments
"""

import argparse

import torch_geometric
from torch_geometric.loader import DataLoader

from data.dataset import CellEncodedDataset

DATA_PATH = r'./datasets'

def parse_args():
    parser = argparse.ArgumentParser(description='Apply Cell Encoding to a dataset.')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='dataset name (default: MUTAG)')
    parser.add_argument('--circles', action='store_true', dest="use_circles",
                        help='Attach cells to all circles of size up to k.')
    parser.add_argument('--cliques', action='store_true', dest="use_cliques",
                        help='Attach cells to all cliques of size up to k.')
    parser.add_argument('--k', type=int, default=5,
                        help='Maximum size of the structure to attach cells to.')
    args = parser.parse_args()

    if args.use_cliques and args.use_circles:
        raise ValueError("Cannot use --circles and --cliques at the same time.")

    if not args.use_cliques and not args.use_circles:
        raise ValueError("Need to use one of --circles or --cliques.")

    if args.k < 0:
        raise ValueError("k must be non-negative.")

    return args

def main():
    args = parse_args()
    print(args)

    og_data = torch_geometric.datasets.TUDataset(root=DATA_PATH, name=args.dataset)
    trans_data = CellEncodedDataset(DATA_PATH, args.dataset, args.k, args.use_cliques)

    print(og_data[0].edge_index)
    print(trans_data[0].edge_index)

if __name__ == "__main__":
    main()
