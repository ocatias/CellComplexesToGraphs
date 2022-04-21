"""
For a given config file of parameters, run a gridsearch over them
i.e. run k-fold cross validation for each possible parameter combination

Run with a config file:
python experiments\run_gridsearch.py FILENAME
"""

import yaml
import argparse
import copy

from experiments.run_on_all_folds import run_with_partial_args as run_experiment

def main():
    parser = argparse.ArgumentParser(description='Run a gridsearch.')
    parser.add_argument('config_file', type=argparse.FileType(mode='r'),
                            help='Path to a config file that should be used for this experiment. ')

    args = parser.parse_args()
    config = yaml.load(args.config_file, Loader=yaml.FullLoader)
    # Split into parameters to tune
    tuneable_parameters = {}
    constant_parameters = {}
    for key, value in config.items():

        if key == "config_file":
            continue

        if isinstance(value, list):
            tuneable_parameters["--" + key] = value
        else:
            constant_parameters["--" + key] = value

    tune(constant_parameters, tuneable_parameters)

def tune(args, tuneable_parameters):
    if len(tuneable_parameters) > 0:
        key, values = list(tuneable_parameters.items())[0]
        del tuneable_parameters[key]
        for value in values:
            args[key] = value
            tune(args, copy.deepcopy(tuneable_parameters))
    else:
        run_experiment(args)

if __name__ == "__main__":
    main()
