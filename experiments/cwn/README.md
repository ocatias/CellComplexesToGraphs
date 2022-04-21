# CW Networks

Implementation of CW Networks, code originally from ![Bodnar et al.](https://github.com/twitter-research/cwn).
Note that this code will only work on GPUs older than the RTX 3000 series, due to driver issues.

## Installation
We use `Python 3.8` and `PyTorch 1.7.0` on `CUDA 10.2` for this project.
Please open a terminal window and follow these steps to prepare the virtual environment needed to run any experiment.

Create the environment:
```shell
conda create --name cwn python=3.8
conda activate cwn
conda install pip # Make sure the environment pip is used
```

Install dependencies:
```shell
conda install -y pytorch=1.7.0 torchvision cudatoolkit=10.2 -c pytorch
sh pyG_install.sh cu102
sh graph-tool_install.sh
pip install -r requirements.txt
```

Add the directories to your pythonpath. Let `@PATH` be the path to where this repository is stored (not the path to the `cwn` folder). 
```shell
export PYTHONPATH=$PYTHONPATH:@PATH:@PATH/cwn/
```

### Credits

Original papers behind this code

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
