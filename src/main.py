#%%
from networks import AutoencoderNet
from loaders import ConfigLoader

import coloredlogs, logging
coloredlogs.install()

import shutil
import argparse
import os

from typing import List

# https://stackoverflow.com/questions/38834378/path-to-a-directory-as-argparse-argument
def file_path(string):
  if os.path.isfile(string):
    return string
  else:
    raise NotADirectoryError(string)

logging.info("----------------------------------")
logging.info("- Welcome to BioP SeSu Lotra DNN -")
logging.info("----------------------------------")

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('--cfg', type=file_path, required=True, 
  help='The main config yaml file.')
parser.add_argument('--env', type=file_path, default='src/yaml/env.yml',
  help='The environment yaml file.')

args = parser.parse_args()

run_cfg_path = args.cfg
env_cfg_path = args.env

# Load configs
env_cfg = ConfigLoader().from_file(env_cfg_path)
run_cfg = ConfigLoader().from_file(run_cfg_path)

# copy the configs to dir for documentation / remembering
rcfg_fn = os.path.split(run_cfg_path)[-1]
ecfg_fn = os.path.split(env_cfg_path)[-1]
model_path = run_cfg['model_path']

if not os.path.exists(model_path):
  os.makedirs(model_path)
  logging.info(f'Created model path: {model_path}')
else:
  logging.warning(f'Model path exists already: {model_path}')

logging.info(f'Copy {run_cfg_path} to {model_path}/{rcfg_fn}')
logging.info(f'Copy {env_cfg_path} to {model_path}/{ecfg_fn}')

shutil.copy(run_cfg_path, f'{model_path}/{rcfg_fn}')
shutil.copy(env_cfg_path, f'{model_path}/{ecfg_fn}')

net = AutoencoderNet(env_cfg, run_cfg)
net.wave_train_test()