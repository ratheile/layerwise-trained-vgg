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
    print("Hello from stackable network")
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


class Autoencoder(nn.Module):

  # upstream_layers: nn.Sequential = None
  # encoder: nn.Sequential = None
  # decoder: nn.Sequential = None

  def __init__(self, color_channels=3):
    # super(Autoencoder, self).__init__()
    super().__init__()
    print("Hello from autoencoder")

    # Conv2d:      b,1c,w,h       -->  b,8c,w,h
    # MaxPool2d:   b,8c,w,h       -->  b,8c,w/2,h/2
    # Conv2d:      b,8c,w/2,h/2   -->  b,16c,w/2,h/2
    # MaxPool2d:   b,16c,w/2,h/2  -->  b,16c,w/4,h/4
    self.upstream_layers = nn.Sequential( 
      nn.Conv2d(in_channels=color_channels, out_channels=color_channels*8, kernel_size=3, stride=1, padding=1),   
      nn.BatchNorm2d(num_features=color_channels*8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
      nn.ReLU(True),
      nn.MaxPool2d(kernel_size=2),  
    )

    self.encoder = nn.Sequential(
      nn.Conv2d(in_channels=color_channels*8, out_channels=color_channels*16, kernel_size=3, stride=1, padding=1), 
      nn.BatchNorm2d(num_features=color_channels*16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
      nn.LeakyReLU(True),
      nn.MaxPool2d(kernel_size=2)   
    )

    # Interpolate:  b,16c,w/4,h/4  -->  b,16c,w/2,h/2
    # Conv2d:       b,16c,w/2,h/2  -->  b,8c,w/2,h/2
    # Interpolate:  b,8c,w/2,h/2   -->  b,8c,w,h
    # Conv2d:       b,8c,w,h       -->  b,1c,w,h
    self.decoder = nn.Sequential(
      Interpolate(),                
      nn.Conv2d(in_channels=color_channels*16, out_channels=color_channels*8, kernel_size=3, stride=1, padding=1),       
      nn.BatchNorm2d(num_features=color_channels*8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
      nn.ReLU(True),
      Interpolate(),                
      nn.Conv2d(in_channels=color_channels*8, out_channels=color_channels*1, kernel_size=3, stride=1, padding=1),   
      nn.BatchNorm2d(num_features=color_channels*1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
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

  # supervision: nn.Sequential = None

  def __init__(self, color_channels):
    super().__init__(color_channels)
    # Autoencoder.__init__(self, color_channels=color_channels)
    # StackableNetwork.__init__(self)
    print("Hello from supervised autoencoder")

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

  N: int = 0 # # of input elements
  M: int = 0 # # of output elements
  # P_st: Paramter = None # Sparse Tesnsor

  in_shape: Tuple[int, int, int] = None
  out_shape: Tuple[int, int, int] = None

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

    
class NetworkStack(nn.Module):

  networks: List[StackableNetwork] = None
  networks_sn: nn.ModuleList = None # StackableNetworks
  networks_maps: nn.ModuleList = None # upstream maps
  map_to_input: Callable[[Tensor], Tensor] 
  
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
    for i in range(len(self.networks) - 1):
      with no_grad():
        net, map_module = self.networks[i]
        x = net.calculate_upstream(x)
        x = map_module.forward(x)
    return x

  
  def forward(self, x):
    x = self.upwards(x)
    top_net, _ = self.networks[-1]
    decoding, prediction = top_net.forward(x)
    return decoding, prediction
    