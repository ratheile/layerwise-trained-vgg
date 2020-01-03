source $HOME/miniconda3/etc/profile.d/conda.sh
cd /cluster/home/ratheile/scaling-spoon-vgg-kaan/

conda env create -f environment.yml
conda activate deeplearning
python src/main.py --cfg "$@"
