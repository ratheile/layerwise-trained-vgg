#%%
from networks import AutoencoderNet
from loaders import ConfigLoader

import coloredlogs, logging
coloredlogs.install()

import shutil
import os

logging.info("----------------------------------")
logging.info("- Welcome to BioP SeSu Lotra DNN -")
logging.info("----------------------------------")

run_cfg_path =  'src/yaml/nets/vgg_B_50ep.yml'
env_cfg_path = 'src/yaml/env.yml'

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