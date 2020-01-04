from loaders import ConfigLoader
import argparse
import os

import yaml
from yaml import FullLoader

from itertools import product

import logging, coloredlogs
coloredlogs.install()

from typing import List

# https://stackoverflow.com/questions/38834378/path-to-a-directory-as-argparse-argument
def file_path(path):
  if os.path.isfile(path):
    return path
  else:
    raise NotADirectoryError(path)

def dir_path(path):
  if os.path.isdir(path):
    return path
  else:
    raise NotADirectoryError(path)

def mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path)
    logging.info(f'Created experiment path: {path}')
  else:
    logging.warning(f'Model path exists already: {path}')

def load_yml(path):
  if os.path.isfile(patch_path):
    with open(patch_path) as patch_file:
      data_dict = yaml.load(patch_file, Loader=FullLoader)
  return data_dict

def save_yml(data_dict, path):
  with open(path, 'w', encoding = "utf-8") as yaml_file:
    dump = yaml.dump(data_dict)
    yaml_file.write(dump)

def patch_to_cfgs(cfg_path:str, patch_path: str) -> List[ConfigLoader]:
  patch = load_yml(patch_path)
  experiment_base = patch['experiment_name']
  out_folder = patch['yml_output_folder']
  mkdir(out_folder)

  for key, values in patch['run_cfg'].items():
    for value in values:
      cfg = ConfigLoader().from_file(cfg_path, suppress_print=True)
      cfg[key] = value
      assert cfg[key] == value
      param_name = key.replace('/','_').split('_')
      param_name = ''.join([s[0] for s in param_name])
      str_value = str(value).replace('.', '_')

      experiment_name = f'{experiment_base}_{param_name}={str_value}'
      cfg['model_path'] = f'trained_models/{experiment_name}'
      out_path = f'{out_folder}/{experiment_name}.yml'

      logging.info(f'Saving param: {key} value: {value} to {out_path}')
      save_yml(cfg.env, out_path)
      
def permute_patch_to_cfgs(cfg_path:str, patch_path: str) -> List[ConfigLoader]:
  patch = load_yml(patch_path)
  experiment_base = patch['experiment_name']
  out_folder = patch['yml_output_folder']
  mkdir(out_folder)

  keys = []
  values = []
  for key, vals in patch['run_cfg'].items():
    keys += [key]
    values.append(vals)

  permutations = list(product(*values))
  
  # iterate over all permutation tuples
  for tple in permutations:
    fn_prefix = ''
    cfg = ConfigLoader().from_file(cfg_path, suppress_print=True)
    for id_k, key in enumerate(keys):
      cfg[key] = tple[id_k]
      assert cfg[key] == tple[id_k]

      param_name = key.replace('/','_').split('_')
      param_name = ''.join([s[0] for s in param_name])
      str_value = str(tple[id_k]).replace('.', '_')
      fn_prefix += f'_{param_name}={str_value}'

    experiment_name = f'{experiment_base}{fn_prefix}'
    cfg['model_path'] = f'trained_models/{experiment_name}'
    out_path = f'{out_folder}/{experiment_name}.yml'

    logging.info(f'Saving tuple: {tple} to {out_path}')
    save_yml(cfg.env, out_path)

if __name__ == "__main__":
  # Parse arguments
  parser = argparse.ArgumentParser()

  parser.add_argument('--cfg', type=file_path, required=True, 
  help='The main config yaml file to modify.')
  parser.add_argument('--hparams', type=file_path, nargs='?',
    help='The hyperparam tuning yaml.')
  parser.add_argument('-P', action='store_true', help='Activate permutation mode')
    
  args = parser.parse_args()

  run_cfg_path = args.cfg
  patch_path = args.hparams
  permute = args.P

  if permute:
    permute_patch_to_cfgs(run_cfg_path, patch_path)
  else:
    patch_to_cfgs(run_cfg_path, patch_path)