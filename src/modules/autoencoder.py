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
  

class SidecarAutoencoder(nn.Module):
  def __init__(self, 
  main_network_layer: List[nn.Module], 
  channels:int,
  dropout: int):

    super().__init__()

    bn_args = { "eps":1e-05, "momentum":0.1, "affine":True, "track_running_stats":True }
    c2d_args = { "kernel_size":3, "stride":1, "padding":1 }
    channel_mult = 2

    # Conv2d:      b,1c,w,h       -->  b,8c,w,h
    # MaxPool2d:   b,8c,w,h       -->  b,8c,w/2,h/2
    # Conv2d:      b,8c,w/2,h/2   -->  b,16c,w/2,h/2
    # MaxPool2d:   b,16c,w/2,h/2  -->  b,16c,w/4,h/4
    self.upstream_layers = nn.Sequential(*main_network_layer)

    self.encoder = nn.Sequential(
      nn.Conv2d(in_channels=channels, out_channels=channels*(channel_mult**1), **c2d_args), 
      nn.BatchNorm2d(num_features=channels*(channel_mult**1), **bn_args),
      nn.Dropout(dropout),
      nn.LeakyReLU(True),
      nn.MaxPool2d(kernel_size=2)   
    )

    # Interpolate:  b,16c,w/4,h/4  -->  b,16c,w/2,h/2
    # Conv2d:       b,16c,w/2,h/2  -->  b,8c,w/2,h/2
    # Interpolate:  b,8c,w/2,h/2   -->  b,8c,w,h
    # Conv2d:       b,8c,w,h       -->  b,1c,w,h
    self.final_conv2d = nn.Conv2d(in_channels=channels*8, out_channels=channels*1, **c2d_args)

    self.decoder = nn.Sequential(
      Interpolate(),                
      nn.Conv2d(in_channels=channels*16, out_channels=channels*8, **c2d_args),       
      nn.BatchNorm2d(num_features=channels*8, **bn_args),
      nn.LeakyReLU(True),
      Interpolate(),                
      self.final_conv2d,   
      nn.BatchNorm2d(num_features=channels*1, **bn_args),
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
  

class SupervisedSidecarAutoencoder(SidecarAutoencoder):

  def __init__(self, 
  main_network_layer: List[nn.Module], 
  channels:int,
  dropout: int):

    super().__init__(
      main_network_layer,
      channels,
      dropout
    )

    fc_layer_size = 16*8*8*channels
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


class RandomMap(nn.Module):

  requires_training: bool = False

  def __init__(self, 
    in_shape: Tuple[int, int, int], 
    out_shape: Tuple[int, int, int]
  ):
    super().__init__()
    self.N:int = reduce(lambda a,b: a*b, in_shape)
    self.M:int = reduce(lambda a,b: a*b, out_shape)
    
    self.out_shape = out_shape
    self.in_shape = in_shape

    P = permutation(np.eye(self.N))
    P_t = torch.from_numpy(P.astype(np.float32))

    # self.P_st = sparse.IntTensor(P_t)
    self.P_st = torch.nn.Parameter(P_t)
    
    

  def forward(self, x):
    assert len(x.shape) is 4
    x_perm = x.view(x.size(0), -1) @ self.P_st
    x = torch.narrow(x_perm, dim=1, start=0, length=self.M)
    return x.view([x.size(0), *self.out_shape])


class SidecarMap(nn.Module):

  requires_training: bool = False
  
  def __init__(self, main_network_layer: List[nn.Module]):
    super().__init__()
    self.function = nn.Sequential(*main_network_layer)

  def forward(self, x):
    x = self.function(x)
    return x
    


class InterpolationMap(nn.Module):

  requires_training: bool = False
  
  def __init__(self):
    pass

  def forward(self, x):
    pass


class ConvMap(nn.Module):

  requires_training: bool = True

  def __init__(self,
    in_shape: Tuple[int, int, int],
    out_shape: Tuple[int, int, int]
  ):
    super().__init__() 
    
    bn_args = { "eps":1e-05, "momentum":0.1, "affine":True, "track_running_stats":True }
    c2d_args = { "kernel_size":3, "stride":1, "padding":1 }

    self.map = nn.Sequential(
      Interpolate(),
      nn.Conv2d(in_channels=in_shape[0], out_channels=out_shape[0], **c2d_args), 
      nn.BatchNorm2d(num_features=out_shape[0], **bn_args),
      nn.LeakyReLU(True)
    )

  def forward(self, x):
    x = self.map(x)
    return x


class DecoderMap(nn.Module):

  requires_training: bool = True


  def __init__(self, trained_net: Autoencoder):
    super().__init__()
    self.initialized: bool = False

    tensor = trained_net.final_conv2d
    bn_args = { "eps":1e-05, "momentum":0.1, "affine":True, "track_running_stats":True }
    
    self.map = nn.Sequential(
      Interpolate(),
      nn.Conv2d(
        in_channels=tensor.in_channels,
        out_channels=tensor.out_channels,
        kernel_size=tensor.kernel_size,
        stride=tensor.stride,
        padding=tensor.padding
      ), 
      nn.BatchNorm2d(num_features=tensor.out_channels, **bn_args),
      nn.Dropout(0.3),
      nn.LeakyReLU(True)
    )

    # have to store these in an array to not make nn.Modle add
    # these params to the self.parameters() list,
    # otherwise they are considedered for optimization which is bad
    self.fixed_params = {'tensor': tensor}

  def forward(self, x):
    if not self.initialized:
      self.initialized = True
      tensor = self.fixed_params['tensor']
      state = tensor.state_dict()
      self.map[1].load_state_dict(state)
      pass # get map from trained_net
    x = self.map(x)
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
    