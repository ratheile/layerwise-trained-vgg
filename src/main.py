#%%
from networks import AutoencoderNet
from loaders import ConfigLoader

import coloredlogs, logging
coloredlogs.install()


logging.info("----------------------------------")
logging.info("- Welcome to BioP SeSu Lotra DNN -")
logging.info("----------------------------------")


env_cfg = ConfigLoader().from_file('src/yaml/env.yml')
run_cfg = ConfigLoader().from_file('src/yaml/nets/net_template.yml')
net = AutoencoderNet(env_cfg, run_cfg)
net.train_test()