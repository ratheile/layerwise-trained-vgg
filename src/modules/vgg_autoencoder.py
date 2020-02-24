
r"""
vgg_autoencoder.py
==================

This module contains all autoencoders required to train a
VGG network horizontally.

.. autosummary::
  modules.SidecarAutoencoder
  modules.SupervisedSidecarAutoencoder
"""
import torch
from torch import nn, Tensor

from .interpolate import Interpolate
from .fcview import FCView

from typing import List, Callable, Tuple

encoders_dict = {

  'A': lambda dropout, num_channels, channel_mult, c2d_args: nn.Sequential(
    nn.Dropout(dropout),
    
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=num_channels, out_channels=num_channels*(channel_mult), **c2d_args), 
    nn.BatchNorm2d(num_features=num_channels*(channel_mult)),
    nn.LeakyReLU(True),   
    nn.Dropout(dropout),

    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=num_channels*(channel_mult), out_channels=num_channels*(channel_mult**2), **c2d_args), 
    nn.BatchNorm2d(num_features=num_channels*(channel_mult**2)),
    nn.LeakyReLU(True),
    nn.Dropout(dropout),
  ), 

  'B': lambda dropout, num_channels, channel_mult, c2d_args: nn.Sequential(
    nn.Dropout(dropout),
    
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=num_channels, out_channels=num_channels*(channel_mult), **c2d_args), 
    nn.BatchNorm2d(num_features=num_channels*(channel_mult)),
    nn.LeakyReLU(True),
    nn.Dropout(dropout),   
  ),

  'C': lambda dropout, num_channels, channel_mult, c2d_args: nn.Sequential(
    nn.Dropout(dropout),
    
    nn.MaxPool2d(kernel_size=2, stride=2),
    # Reduce channel sizes instead of increasing them in B here:
    nn.Conv2d(in_channels=num_channels, out_channels=int(num_channels/(channel_mult)), **c2d_args), 
    nn.BatchNorm2d(num_features=int(num_channels/(channel_mult))),
    nn.LeakyReLU(True),
    nn.Dropout(dropout),   
  )
}

decoders_dict = {

  'A': lambda dropout, num_channels, channel_mult, c2d_args, in_channels: nn.Sequential(
    Interpolate(),                
    nn.Conv2d(in_channels=num_channels*(channel_mult**2), out_channels=num_channels*(channel_mult), **c2d_args),       
    nn.BatchNorm2d(num_features=num_channels*(channel_mult)),
    nn.LeakyReLU(True),
    nn.Dropout(dropout),

    Interpolate(),                
    nn.Conv2d(in_channels=num_channels*(channel_mult), out_channels=num_channels, **c2d_args),       
    nn.BatchNorm2d(num_features=num_channels),
    nn.LeakyReLU(True),
    nn.Dropout(dropout),

    nn.Conv2d(in_channels=num_channels, out_channels=in_channels, **c2d_args),
    nn.BatchNorm2d(num_features=in_channels),
    nn.Sigmoid()
  ),

  'B': lambda dropout, num_channels, channel_mult, c2d_args, in_channels: nn.Sequential(
    Interpolate(),                
    nn.Conv2d(in_channels=num_channels*(channel_mult), out_channels=num_channels, **c2d_args),       
    nn.BatchNorm2d(num_features=num_channels),
    nn.LeakyReLU(True),
    nn.Dropout(dropout),

    nn.Conv2d(in_channels=num_channels, out_channels=in_channels, **c2d_args),
    nn.BatchNorm2d(num_features=in_channels),
    nn.Sigmoid()
  ),

  'C': lambda dropout, num_channels, channel_mult, c2d_args, in_channels: nn.Sequential(
    Interpolate(),                
    nn.Conv2d(in_channels=int(num_channels/(channel_mult)), out_channels=num_channels, **c2d_args),       
    nn.BatchNorm2d(num_features=num_channels),
    nn.LeakyReLU(True),
    nn.Dropout(dropout),

    nn.Conv2d(in_channels=num_channels, out_channels=in_channels, **c2d_args),
    nn.BatchNorm2d(num_features=in_channels),
    nn.Sigmoid()
  )
}
class SidecarAutoencoder(nn.Module):
  r"""
  This autoencoder is a "sidecar" to a deeper network. It takes a slice of
  the main network (main_network_layer) and trains this layer isolated from the
  other parts of the main network.

  Different autoencoder types are supported (A-C).
  """

  def __init__(self, 
  main_network_layer: List[nn.Module], 
  img_size: int,
  channels: Tuple[int, int],
  dropout: float,
  kernel_size: int,
  encoder_type:str):

    super().__init__()

    c2d_args = { 
      "kernel_size":kernel_size,
      "stride":1,
      "padding":int(kernel_size / 2)
    }

    channel_mult = 2
    in_channels = channels[0]
    out_channels = channels[-1]

    self.encoder_type = encoder_type
    self.upstream_layers = nn.Sequential(*main_network_layer)

    # First dropout layer is for upstream training, but should not be there in upstream processing
    self.encoder = encoders_dict[encoder_type](
      dropout, out_channels,
      channel_mult, c2d_args
    )

    self.decoder = decoders_dict[encoder_type](
      dropout, out_channels,
      channel_mult, c2d_args, in_channels
    )

    self.img_size = img_size
    self.channel_mult = channel_mult
    self.num_channels = out_channels

  def bottleneck_size(self) -> int:
    if self.encoder_type == 'A':
      return int((self.img_size/4)**2 * (self.num_channels * self.channel_mult**2))
    elif self.encoder_type == 'B':
      return int((self.img_size/2)**2 * (self.num_channels * self.channel_mult))
    elif self.encoder_type == 'C':
      return int((self.img_size/2)**2 * (self.num_channels / self.channel_mult))

      
  def calculate_upstream(self, x):
    x = self.upstream_layers(x)
    return x

  def forward(self, x):
    x = self.upstream_layers(x)
    x = self.encoder(x)
    x = self.decoder(x)
    return x
  

class SupervisedSidecarAutoencoder(SidecarAutoencoder):
  r"""
  The supervision extension of the sidecar autoencoder.
  """
  def __init__(self, 
  main_network_layer: List[nn.Module], 
  img_size: int,
  channels: Tuple[int, int],
  dropout: float,
  kernel_size:int,
  encoder_type:int,
  num_classes: int):

    super().__init__(
      main_network_layer,
      img_size,
      channels,
      dropout,
      kernel_size,
      encoder_type
    )

    fc_layer_size = self.bottleneck_size()
    self.supervision = nn.Sequential(
      FCView(),
      nn.Linear(in_features=fc_layer_size, out_features=512),
      nn.ReLU(True),
      nn.Dropout(dropout),
      nn.Linear(in_features=512, out_features=num_classes)
      # last non-linearity should (and is) included in the loss function
    )

  def forward(self, x):
    upstream = self.upstream_layers(x)
    encoding = self.encoder(upstream)
    prediction = self.supervision(encoding)
    decoding = self.decoder(encoding)
    return decoding, prediction

