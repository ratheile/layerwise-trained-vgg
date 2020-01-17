r"""
layer_training_def.py
=====================
Wrappers / Definitions needed in our network.

.. autosummary::
  networks.LayerTrainingDefinition
  LayerType
"""
from torch import nn
from dataclasses import dataclass
from torch.optim import Optimizer
from modules import NetworkStack
from enum import Enum

class LayerType(Enum):
  r"""
  Enum that describes different layer types.

  - VGGlinear : the last layer of our VGG network
  - Stack: any other layer trained by a supervised autoencoder
  """
  VGGlinear = 0
  Stack = 1

@dataclass
class LayerTrainingDefinition:
  r"""
  This class is a wrapper for all objects that
  are needed in a layer training process.

  The property stack contains a reference to all
  the previous layers such that the upstream
  (representation of x before this layer)
  can be calculated.
  """

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

