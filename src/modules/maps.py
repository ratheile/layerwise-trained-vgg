r"""
maps.py
=======
The term map in this project is a nn.module that maps data from one horizontal
layer to the next. It is the abstract concept of coupling such individually 
trained layers into one bigger (deeper) network.

Many different maps are collected in this module. It one can check in the config
yaml to see which of these maps is used in a specific training instance.

.. autosummary::
  modules.RandomMap
  modules.SidecarMap
  modules.ConvMap
  modules.DecoderMap
"""
import torch
from torch import nn, Tensor
from numpy.random import permutation

from .interpolate import Interpolate
from .autoencoder import Autoencoder

from functools import reduce
from typing import List, Callable, Tuple

class RandomMap(nn.Module):
  r"""
  This map randomly permutes the input and maps it
  to the output shape.
  """

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
  r"""
  This map takes any set of
  non-trainable layer from a bigger main network
  and uses it as a map function.

  Mainly used for maxpool layers.
  """

  requires_training: bool = False
  
  def __init__(self, main_network_layer: List[nn.Module]):
    super().__init__()
    self.function = nn.Sequential(*main_network_layer)

  def forward(self, x):
    x = self.function(x)
    return x


class ConvMap(nn.Module):
  r"""
  Trainable map that is a one layer convolution
  to map from input to output dimension.
  """

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
  r"""
  This map takes the last layer of a horizontal autoencoder and
  uses it as an upstream mapping initialization.
  No scaling is needed since this layer has
  the exact dimensions we need by design.
  
  Other than that, it is similar to ConvMap.
  """

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
    
