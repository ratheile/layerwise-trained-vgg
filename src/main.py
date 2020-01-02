#%%
from networks import AutoencoderNet
from loaders import ConfigLoader

import coloredlogs, logging
coloredlogs.install()

import shutil
import argparse
import yaml
from yaml import FullLoader
import os

from typing import List

# https://stackoverflow.com/questions/38834378/path-to-a-directory-as-argparse-argument
def file_path(string):
  if os.path.isfile(string):
    return string
  else:
    raise NotADirectoryError(string)

def values_to_cfgs(env_cfg_path:str, key:str, values):
  env_cfgs = []
  # for v in values:
  #   env_cfg = ConfigLoader().from_file(env_cfg_path)
  #   env_cfgs += [env_cfg]
  print(key, values)
  return env_cfgs

def patch_to_cfgs(env_cfg_path:str, patch_path: str) -> List[ConfigLoader]:
  env_cfgs = []
  if os.path.isfile(patch_path):
    with open(patch_path) as patch_file:
      patch = yaml.load(patch_file, Loader=FullLoader)
      for key, value in patch.items():
        if key == 'layers':
          print(value)
          for id_l, layer_dict in enumerate(value):
            for l_key, l_value in layer_dict.items():
              env_cfgs += values_to_cfgs(None, 
              f'layer/{id_l}/{l_key}', l_value)
        else:
          env_cfgs += values_to_cfgs(None, key, value)
  return []

logging.info("----------------------------------")
logging.info("- Welcome to BioP SeSu Lotra DNN -")
logging.info("----------------------------------")

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('--cfg', type=file_path, required=True, 
  help='The main config yaml file.')
parser.add_argument('--env', type=file_path, default='src/yaml/env.yml',
  help='The environment yaml file.')
parser.add_argument('--hparams', type=file_path, nargs='?',
  help='The hyperparam tuning yaml.')

args = parser.parse_args()

run_cfg_path = args.cfg
env_cfg_path = args.env
hparams_path = args.hparams

if hparams_path is not None:
  env_cfg = ConfigLoader().from_file(env_cfg_path)
  run_configs: List[ConfigLoader] = patch_to_cfgs(run_cfg_path, hparams_path)
  for run_cfg in run_configs:
    print(run_cfg)

else:
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
  net.train_test()