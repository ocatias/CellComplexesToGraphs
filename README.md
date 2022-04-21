# Reducing Learning on Cell Complexes to Graphs

Code repository for our paper ['Reducing Learning on Cell Complexes to Graphs'](https://openreview.net/pdf?id=HKUxAE-J6lq) accepted at the 'Geometrical and Topological Representation Learning' workshop at ICLR 2022.

## How to install
We recommend the use of conda, as that makes the installation of graph-tool very straightforward.
- Clone this repository and open it
- Create a new conda environment `conda create -n myenv python=3.9`
- Activate environment `conda activate myenv`
- Install [pytorch](https://pytorch.org/)
- Install [pytorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- Install graphtool `conda install -c conda-forge graph-tool`
- Install other requirements `pip install -r requirements.txt`

Note that you need a WandB account to run any experiments that make use of Bayesian optimization.

## Getting started
First activate the environment  `conda activate myenv` and add the code directory to the paths variable `set PYTHONPATH=$PathTo\CellComplexesToGraphs` where $PathTo is the path to the directory containing this repository.

### Experiments on NCI1, NCI109 and Proteins
Note that for every wandb sweep you run you need to tweak the .yaml file to point towards your python executable, for example replacing `/home/.../miniconda3/envs/my_env/bin/python` by the path to your python binary.

You can run the experiments by first starting a sweep for your dataset with one of those commands:
```
wand sweep experiments_configs/NCI1_rings_sweep.yaml
wand sweep experiments_configs/NCI109_rings_sweep.yaml
wand sweep experiments_configs/protein_rings_sweep.yaml
```
Aftwards, you need to join at least one agent. The explicit command command to join an agent will be shown in the terminal when executing the previous command, or can be seen in your sweep controls. You can see the results of the sweep on the WandB dashboard. After a suitable amount of parameter combinations (we recommend 20) you should get a good validation accuracy (Final/Best_avg_acc).

### Ablation on NCI1
In what follows `$workers` are the number of workers that will train models at the same time. `$candidates` are the number of parameter combinations to try per fold and `$repeats` is the number of times to train the model with the best parameters per fold. Results should be stored in the `/results/` directory.

- GIN: `python experiments/run_clean_cv.py  -grid experiments_configs/tu_clean_grid_just_gin.yaml -dataset NCI1 --workers $workers --candidates $candidates --repeats $repeats`
- GIN+CRE: `python experiments/run_clean_cv.py  -grid experiments_configs/tu_clean_grid.yaml.yaml -dataset NCI1 --workers $workers --candidates $candidates --repeats $repeats`
- WL-SP (1 iter): `python experiments/run_clean_cv.py  -grid experiments_configs/wl-sp_1iter_grid.yaml -dataset NCI1 --workers $workers --candidates $candidates --repeats $repeats --kernel`
- WL-SP (2 iter): `python experiments/run_clean_cv.py  -grid experiments_configs/wl-sp_2iter_grid.yaml -dataset NCI1 --workers $workers --candidates $candidates --repeats $repeats --kernel`
- WL-SP (1 iter) + CRE:  `python experiments/run_clean_cv.py  -grid experiments_configs/wl-sp_1iter_cre_grid.yaml -dataset NCI1 --workers $workers --candidates $candidates --repeats $repeats --kernel`
- WL-SP (2 iter) + CRE:  `python experiments/run_clean_cv.py  -grid experiments_configs/wl-sp_2iter_cre_grid.yaml -dataset NCI1 --workers $workers --candidates $candidates --repeats $repeats --kernel`
- WL-ST: `python experiments/run_clean_cv.py  -grid experiments_configs/wl-st_grid.yaml -dataset NCI1 --workers $workers --candidates $candidates --repeats $repeats --kernel`
- WL-ST + CRE:  `python experiments/run_clean_cv.py  -grid experiments_configs/wl-st_cre_grid.yaml   -dataset NCI1 --workers $workers --candidates $candidates --repeats $repeats --kernel`

For CIN, go to `/experiments/cwn/` and follow the setup guide. Then run
`python exp/run_clean_cv.py grid exp/tuning_configurations/NCI1_clean_cv.yaml -dataset NCI1 --workers $workers --candidates $candidates --repeats $repeats`
Please note that due to pytorch compatibility issues you will need a GPU that is older than the RTX 30xx series.


### ogb-molhiv
First tune the parameters with
`wandb sweep experiments_configs/molhiv_sweep.yaml `
Remember to join agents (see above). Afterwards select the parameters with the highest validation accuracy from WandB and rerun them again (command can be found in WandB) to get test results.

## How to Cite
If you make use of our code or ideas please cite

```
@inproceedings{jogl2022reducing,
  title={Reducing Learning on Cell Complexes to Graphs},
  author={Jogl, Fabian and Thiessen, Maximilian and G{\"a}rtner, Thomas},
  booktitle={ICLR 2022 Workshop on Geometrical and Topological Representation Learning},
  year={2022}
}
```

## Credits
The code in this repository is based on

```
@inproceedings{pmlr-v139-bodnar21a,
  title = 	 {Weisfeiler and Lehman Go Topological: Message Passing Simplicial Networks},
  author =       {Bodnar, Cristian and Frasca, Fabrizio and Wang, Yuguang and Otter, Nina and Montufar, Guido F and Li{\'o}, Pietro and Bronstein, Michael},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {1026--1037},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
}
```

```
@inproceedings{neurips-bodnar2021b,
  title={Weisfeiler and Lehman Go Cellular: CW Networks},
  author={Bodnar, Cristian and Frasca, Fabrizio and Otter, Nina and Wang, Yu Guang and Li{\`o}, Pietro and Mont{\'u}far, Guido and Bronstein, Michael},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {34},
  year={2021}
}
```

```
@inproceedings{
xu2018how,
title={How Powerful are Graph Neural Networks?},
author={Keyulu Xu and Weihua Hu and Jure Leskovec and Stefanie Jegelka},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=ryGs6iA5Km},
}
```

```
@article{hu2020ogb,
  title={Open Graph Benchmark: Datasets for Machine Learning on Graphs},
  author={Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
  journal={arXiv preprint arXiv:2005.00687},
  year={2020}
}
```
