import unittest   # The test framework
import torch
import numpy as np

from networks import AutoencoderNet, LayerTrainingDefinition, cfg_to_network

from io import StringIO 
import yaml

from loaders import ConfigLoader

from typing import List, IO

class Test_Persistence(unittest.TestCase):


  def test_save(self):
    cfg = ConfigLoader().from_file('yaml/env.yml')
    net = AutoencoderNet(cfg['datasets/cifar10/path'])
    layers: List[LayerTrainingDefinition] = net.layer_configs 
    net.save_layer(layers[0].model, self.file)


  def test_restore(self):
    pass

  
  def test_cfg_to_network(self):
    rcfg = ConfigLoader().from_file('src/yaml/nets/net_template.yml')
    gcfg = ConfigLoader().from_file('src/yaml/env.yml')
    net = cfg_to_network(gcfg, rcfg)
    print(net)

  def test_config_loader_from_file(self):
    loader = ConfigLoader().from_file('yaml/env_template.yml')
    self.assertEqual(loader['datasets/cifar10/path'], '<path>')


  def test_config_loader_from_str(self):
    document = """
      a: 1
      b:
        c: 3
        d: 4
    """

    loader = ConfigLoader().from_string(document)
    self.assertEqual(loader['b/c'], 3)
    self.assertEqual(loader['a'], 1)
    self.assertEqual(loader['b']['c'], 3)
    self.assertEqual(loader['b']['d'], 4)
    
    self.assertEqual(loader['b', 'c'], 3)


  def test_array(self):
    document = """
      l:
        - list: str
          attr: 1
        - test: nr
          attr: 2
    """
    loader = ConfigLoader().from_string(document)
    self.assertEqual(loader['l/0/attr'], 1)
    self.assertEqual(loader['l/1/attr'], 2)
    

if __name__ == '__main__':
    unittest.main()