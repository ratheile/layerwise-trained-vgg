r"""
autoencoder.py
==============

First implementation of a horizontal (layerwise) training mechanism of a deep network.
It is fundamentally flawed because the receptive field of this stack does not
increase in depth and therefore it is not possible for the network to learn
complex features in images.

This happens because the upstream maps the data back to the input size of this
autoencoder to make it stackable.

Replaced by SidecarAutoencoder which trains a narrowing network that does not have
these problems. We keep the code for reasoning / documentation of our mistakes.

.. autosummary::
  modules.Autoencoder
  modules.SupervisedAutoencoder
"""
import torch
from torch import nn, Tensor
from torch.nn import Parameter
from torch import sparse

from .interpolate import Interpolate
from .stackable_network import StackableNetwork
from .fcview import FCView

from typing import List, Callable, Tuple

class Autoencoder(nn.Module, StackableNetwork):
  r"""
  Implementation of an autoencoder used in our stack. This is a horizontal module.
  It is connected to the upper layer via its upstream function.
  """

  def __init__(self, color_channels: int=3, dropout: int=0.3):
    super().__init__()


    bn_args = { "eps":1e-05, "momentum":0.1, "affine":True, "track_running_stats":True }
    c2d_args = { "kernel_size":3, "stride":1, "padding":1 }


    # Conv2d:      b,1c,w,h       -->  b,8c,w,h
    # MaxPool2d:   b,8c,w,h       -->  b,8c,w/2,h/2
    # Conv2d:      b,8c,w/2,h/2   -->  b,16c,w/2,h/2
    # MaxPool2d:   b,16c,w/2,h/2  -->  b,16c,w/4,h/4
    self.upstream_layers = nn.Sequential( 
      nn.Conv2d(in_channels=color_channels, out_channels=color_channels*8, **c2d_args),   
      nn.BatchNorm2d(num_features=color_channels*8, **bn_args),
      nn.Dropout(dropout),
      nn.LeakyReLU(True),
      nn.MaxPool2d(kernel_size=2),  
    )

    self.encoder = nn.Sequential(
      nn.Conv2d(in_channels=color_channels*8, out_channels=color_channels*16, **c2d_args), 
      nn.BatchNorm2d(num_features=color_channels*16, **bn_args),
      nn.Dropout(0.3),
      nn.LeakyReLU(True),
      nn.MaxPool2d(kernel_size=2)   
    )

    # Interpolate:  b,16c,w/4,h/4  -->  b,16c,w/2,h/2
    # Conv2d:       b,16c,w/2,h/2  -->  b,8c,w/2,h/2
    # Interpolate:  b,8c,w/2,h/2   -->  b,8c,w,h
    # Conv2d:       b,8c,w,h       -->  b,1c,w,h
    self.final_conv2d = nn.Conv2d(in_channels=color_channels*8, out_channels=color_channels*1, **c2d_args)

    self.decoder = nn.Sequential(
      Interpolate(),                
      nn.Conv2d(in_channels=color_channels*16, out_channels=color_channels*8, **c2d_args),       
      nn.BatchNorm2d(num_features=color_channels*8, **bn_args),
      nn.LeakyReLU(True),
      Interpolate(),                
      self.final_conv2d,   
      nn.BatchNorm2d(num_features=color_channels*1, **bn_args),
      nn.Tanh()
    )

  def calculate_upstream(self, x):
    r"""
    computes the representation of x used in the next layer
    """
    x = self.upstream_layers(x)
    return x

  def forward(self, x):
    x = self.upstream_layers(x)
    x = self.encoder(x)
    x = self.decoder(x)
    return x
  

class SupervisedAutoencoder(Autoencoder):
  r"""
  A supervised extension to the autoencoder class.
  It extends autoencoder with a fully connected module linked to the autoencoder
  bottleneck.
  """

  def __init__(self, color_channels: int):
    super().__init__(color_channels)

    fc_layer_size = 16*8*8*color_channels
    self.supervision = nn.Sequential(
      FCView(),
      nn.Linear(in_features=fc_layer_size, out_features=100),
      nn.Linear(in_features=100, out_features=10),
    )

  def forward(self, x):
    upstream = self.upstream_layers(x)
    encoding = self.encoder(upstream)
    prediction = self.supervision(encoding)
    decoding = self.decoder(encoding)
    return decoding, prediction