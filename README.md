# scaling-spoon
Layerwise Trained Deep Neural Network

# Where is what?

Documentation is in `doc_build/html/index.html`, generated from the docstrings using the recommended python documentation module `sphinx`.

We use yaml files to configure our network (`src/yaml`). Attach them to main with the respective parameters:

  - Environment Config `--env` contains all parameters to run the programs on a specific computer (paths etc.)
  - Run Config `--cfg` contains all parameters to run the network (hyperparameters).

Example:

```
conda activate deeplearning
cd src
python main --cfg=<cfg-path> --env=<env-path>
```

The main programs are `src/main.py` and the utility tool `src/yml_gen.py` to generate permutations of configurations for hyperparameter tuning.

All results are collected in tensorboard

```
tensorboard --logdir=
```

If `store_model` is true, then the trained weights (tensors) are stored to the path given in the config file and can be loaded again by giving the file name to the specific layer.