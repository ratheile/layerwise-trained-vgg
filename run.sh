source $HOME/miniconda3/etc/profile.d/conda.sh
cd /cluster/home/koktay/Grewe/scaling-spoon/

conda env create -f environment.yml
conda activate deeplearning
python src/main.py
