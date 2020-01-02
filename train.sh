#!/bin/bash
source /home/shafall/opt/miniconda3/etc/profile.d/conda.sh
conda activate grewe
python src/main.py --cfg src/yaml/nets/vgg_raffael.yml
