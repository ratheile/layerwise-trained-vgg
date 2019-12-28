import torch
from torch import nn, Tensor
from torch.nn import Parameter
from torch import sparse
from torch import no_grad
from .interpolate import Interpolate
from .fcview import FCView
from typing import List, Callable, Tuple

from functools import reduce
import numpy as np
from numpy.random import permutation

class StackableNetwork(object):

  def __init__(self):
    pass

  """
  Interface, not a Class! Do not implement anything here
  Classes that inherit from StackableNetwork are required,
  (in python by convention, not enforced) to provide an
  implementation of these functions.
  """

  def calculate_upstream(self, previous_network):
    """
    Calculate the output to the point where the
    map function will be applied after.

    Make sure those layers appear in parameters()
    """
    raise "Provide an upstream function"
  

class SidecarAutoencoder(nn.Module):
  def __init__(self, 
  main_network_layer: List[nn.Module], 
  img_size: int,
  channels: Tuple[int, int],
  dropout: float):

    super().__init__()

    bn_args = { "eps":1e-05, "momentum":0.1, "affine":True, "track_running_stats":True }
    c2d_args = { "kernel_size":3, "stride":1, "padding":1 }
    channel_mult = 2
    in_channels = channels[0]
    num_channels = channels[-1]

    # Conv2d:      b,1c,w,h       -->  b,8c,w,h
    # MaxPool2d:   b,8c,w,h       -->  b,8c,w/2,h/2
    # Conv2d:      b,8c,w/2,h/2   -->  b,16c,w/2,h/2
    # MaxPool2d:   b,16c,w/2,h/2  -->  b,16c,w/4,h/4
    self.upstream_layers = nn.Sequential(*main_network_layer)

    self.encoder = nn.Sequential(
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(in_channels=num_channels, out_channels=num_channels*(channel_mult), **c2d_args), 
      nn.BatchNorm2d(num_features=num_channels*(channel_mult), **bn_args),
      nn.Dropout(dropout),
      nn.LeakyReLU(True),   
    )

    # Interpolate:  b,16c,w/4,h/4  -->  b,16c,w/2,h/2
    # Conv2d:       b,16c,w/2,h/2  -->  b,8c,w/2,h/2
    # Interpolate:  b,8c,w/2,h/2   -->  b,8c,w,h
    # Conv2d:       b,8c,w,h       -->  b,1c,w,h

    self.decoder = nn.Sequential(
      Interpolate(),                
      nn.Conv2d(in_channels=num_channels*(channel_mult), out_channels=num_channels, **c2d_args),       
      nn.BatchNorm2d(num_features=num_channels, **bn_args),
      nn.LeakyReLU(True),
      nn.Conv2d(in_channels=num_channels, out_channels=in_channels, **c2d_args),
      nn.BatchNorm2d(num_features=in_channels, **bn_args),
      nn.Sigmoid()
    )

    self.img_size = img_size
    self.channel_mult = channel_mult
    self.num_channels = num_channels

  def bottleneck_size(self) -> int:
    return int((self.img_size/2)**2 * (self.num_channels * self.channel_mult))
      
  def calculate_upstream(self, x):
    x = self.upstream_layers(x)
    return x

  def forward(self, x):
    x = self.upstream_layers(x)
    x = self.encoder(x)
    x = self.decoder(x)
    return x
  

class SupervisedSidecarAutoencoder(SidecarAutoencoder):
  def __init__(self, 
  main_network_layer: List[nn.Module], 
  img_size: int,
  channels: Tuple[int, int],
  dropout: float):

    super().__init__(
      main_network_layer,
      img_size,
      channels,
      dropout
    )

    fc_layer_size = self.bottleneck_size()
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


class SidecarMap(nn.Module):

  requires_training: bool = False
  
  def __init__(self, main_network_layer: List[nn.Module]):
    super().__init__()
    self.function = nn.Sequential(*main_network_layer)

  def forward(self, x):
    x = self.function(x)
    return x


class NetworkStack(nn.Module):

  def __init__(self,
    networks: List[Tuple[StackableNetwork, nn.Module]],
    train_every_pass=False
  ):
    super().__init__()
    self.networks = networks
    self.networks_sn = nn.ModuleList([
      net for net, _ in networks
    ])

    self.networks_maps = nn.ModuleList([
      usmap for _, usmap in networks
    ])

  def upwards(self, x):
    
    n = len(self.networks) 

    # covers case where we only have 2 nets
    if n >= 3:
      for i in range(n-2):
        net, map_module = self.networks[i]
        with no_grad():
          x = net.calculate_upstream(x)
          # in VGG, there is no map module
          if map_module is not None:
            x = map_module.forward(x)
      # end for loop

    if n >= 2:
      # the second last net
      net, map_module = self.networks[n-2] 
      with no_grad():
        x = net.calculate_upstream(x)

      mmc = map_module is not None
      if mmc and map_module.requires_training:
        # map module forward needs to be outside of
        # no_grad environment because of the req.
        # training!
        x = map_module.forward(x)
      else:
        with no_grad():
          if map_module is not None:
            x = map_module.forward(x)
    # end if
    return x

  
  def forward(self, x):
    x = self.upwards(x)
    top_net, _ = self.networks[-1]
    decoding, prediction = top_net.forward(x)
    return decoding, prediction