# scaling-spoon
A Layerwise Trained Deep Neural Network

# How to Install

```
conda env create -f environment.yml
```

Download the datasets (example path in command):
```
python src/download_datasets.py --path="/home/user1/datasets/test"
```

Modify / create your own environment file. Check out the [template](src/yaml/env/env_template.yml).




# How to Train
We use yaml files to configure our network (collected in `src/yaml`). Attach them to main with the respective parameters:

  - Environment Config `--env` contains all parameters to run the programs on a specific computer (paths etc.)
  - Run Config `--cfg` contains all parameters to run the network (hyperparameters).

Run configs are platform agnostic and should work if the environment is configured properly.

Example:
```
conda activate deeplearning
python src/main.py -cfg=src/yaml/<cfg-path> --env=src/yaml/<env-path>
```

The main program is`src/main.py`.

# How to Tune Hyperparameter
We use an utility tool `src/yml_gen.py` to generate configurations for hyperparameter tuning.
It creates a set of `run-cfg.yml` files in a folder specified in the slice.
The slice specifies which parameters should differ from the main configuration file (`--cfg` parameter)
The script then finds all combinations. and writes those to different files

Example: 
```
python src/yml_gen.py -P --cfg=src/yaml/nets/vgg_final.yml --hparam=src/yaml/raffi/hparams/final_tuning.yml
```

# How to Train on ETH Leonhard Cluster
Install miniconda on leonhard if you have not done this so far:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Copy the project folder to the cluster (with `scp` or a similar tool).

We wrote scripts to train on Leonhard. Unfortunately Leonhard uses personalized homes so a few things need to be changed:

Copy these files (the file name should indicate that they are personalized) and change the relevant paths to your account in:
1. bsub_raffi.sh: submits a job
   1. Change the script path to `run_<your script name>.sh`
2. run_raffi.sh: the network bootstrap script
   1. miniconda path
   2. cd to project path
   3. `--env=<your cluster config>`
3. a cluster env.yml ([example](src/yaml/raffi/env/env_cluster_raffi.yml))
   1. dataset path

You can start the training on leonhard now by using the bsub script:

```
./bsub_raffi.sh src/yaml/<path to cfg.yml>

```

or in a for loop to submit tasks at once:

```
for i in src/yaml/raffi/nets_gen/final/*;do ./bsub_raffi.sh "$i";done
```

# How to View Results
All results are collected in tensorboard

```
tensorboard --logdir=runs
```

shows the plots.

## For layers:
(`keys` are from the cfg.yml)

If `pretraining_store` is true, then the trained weights (tensors) are stored to the path 
given in the config file and can be loaded again by giving the file name to the specific layer.

If `pretraining_load=<some filename>`, then the network tries to use this blob to skip training for that layer.
The file has to be in `model_path`. This is also the folder where layers are stored during training.

# Documentation
We tried to keep the code as clean and understandable as possible
Additionally, we generated a small [documentation](scaling_spoon.pdf) to get you started quickly.

It is generated with the `sphinx` framework.