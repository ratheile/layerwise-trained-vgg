from torch import nn
from dataclasses import dataclass
from torch.optim import Optimizer
from modules import NetworkStack

@dataclass
class LayerTrainingDefinition:
  layer_name: str = None
  #config
  num_epochs: int = 0
  pretraining_store: str = None
  pretraining_load: str = None

  # stack including this layer
  stack: NetworkStack = None

  # this layers elements 
  upstream: nn.Module = None
  model: nn.Module = None

  # other vars
  optimizer: Optimizer = None

