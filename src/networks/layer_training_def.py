from torch import nn
from dataclasses import dataclass
from torch.optim import Optimizer
from modules import NetworkStack
from enum import Enum

class LayerType(Enum):
  VGGlinear = 0
  Stack = 1

@dataclass
class LayerTrainingDefinition:

  # naming / ID
  layer_type: LayerType = None
  layer_name: str = None

  # config
  num_epochs: int = 0
  pretraining_store: str = None
  pretraining_load: str = None
  model_base_path:str = None

  # stack including this layer
  stack: NetworkStack = None

  # this layers elements 
  upstream: nn.Module = None
  model: nn.Module = None

  # trainable params
  tp_alpha: nn.Parameter = None

  # other vars
  optimizer: Optimizer = None
  ae_loss_function: str = None

