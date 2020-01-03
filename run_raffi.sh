source $HOME/miniconda3/etc/profile.d/conda.sh
cd /cluster/home/ratheile/scaling-spoon

conda env create -f environment.yml
conda activate deeplearning
python src/main.py --cfg "$@" --env=src/yaml/env_cluster_raffi.yml

