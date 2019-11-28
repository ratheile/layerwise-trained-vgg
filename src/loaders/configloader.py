from __future__ import annotations # for -> ConfigLoader typedef
import yaml
from yaml import FullLoader
import os
from collections.abc import Iterable
import logging

from typing import Dict, Callable

class ConfigLoader(object):

  def from_file(self, env_path: str = 'yaml/env.yml') -> ConfigLoader:
    if os.path.isfile(env_path):
      with open(env_path) as env_file:
        self.env = yaml.load(env_file, Loader=FullLoader)
      logging.info("Config {} is: {}".format(env_path, self.env))
    else:
      logging.warning('Config {} not found'.format(env_path))
    return self

  def from_string(self, data: str) -> ConfigLoader:
    self.env = yaml.load(data, Loader=FullLoader)
    return self
  
  def switch(self, key: str, options: Dict[str, Callable]):
    value = self.__getitem__(key)
    return options[value]()
  
  def __repr__(self):
    return yaml.dump(self.env)

  def __str__(self):
    return yaml.dump(self.env)

  def __getitem__(self, key):
    if isinstance(key, list) or isinstance(key, tuple):
      val = self.env
      for k in key:
        type_check = type(k) is str and isinstance(val, Iterable)
        k = int(k) if str.isdigit(k) else k
        existence_check = len(val) > k if isinstance(val, list) else (k in val)
        if type_check and existence_check:
          val = val[k]
        else:
          raise ValueError("Error : Could not retrieve key {} in  {}".format(k, key))
      return val
    else:
      subkeys = key.split('/')
      if len(subkeys) > 1:
        return self.__getitem__(subkeys)
      else:
        return self.env[key]
  