import torch
from torch import nn, Tensor
from torch.nn import Parameter
from torch import sparse

from .interpolate import Interpolate
from .stackable_network import StackableNetwork
from .fcview import FCView

from typing import List, Callable, Tuple

class Autoencoder(nn.Module, StackableNetwork):

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
    x = self.upstream_layers(x)
    return x

  def forward(self, x):
    x = self.upstream_layers(x)
    x = self.encoder(x)
    x = self.decoder(x)
    return x
  

class SupervisedAutoencoder(Autoencoder):

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