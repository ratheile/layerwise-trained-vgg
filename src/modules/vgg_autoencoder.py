
import torch
from torch import nn, Tensor

from .interpolate import Interpolate
from .fcview import FCView

from typing import List, Callable, Tuple


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

    # First dropout layer is for upstream training, but should not be there in upstream processing
    self.encoder = nn.Sequential(
      nn.Dropout(dropout),
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

