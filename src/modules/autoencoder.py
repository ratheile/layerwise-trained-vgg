from torch import nn, Tensor
from torch import no_grad
from .interpolate import Interpolate
from .fcview import FCView
from typing import List, Callable, Tuple

class StackableNetwork(object):
  """
  Interface, not a Class! Do not implement anything here
  Classes that inherit from StackableNetwork are required,
  (in python by convention, not enforced) to provide an
  implementation of these functions.
  """

  def DEFAULT_MAP_F(self):
    """
    Helper Function: provides a default implementation
    of the upstream mapping function if the user decides
    to not implement something special.
    """
    raise "Provide a default map function"

  def calculate_upstream(self, previous_network):
    """
    Calculate the output to the point where the
    map function will be applied after.
    """
    raise "Provide an upstream function"


class Autoencoder(nn.Module, StackableNetwork):

  upstream_layers: nn.Sequential = None
  encoder: nn.Sequential = None
  decoder: nn.Sequential = None

  def __init__(self, color_channels=3):
    super().__init__()
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
  


def map_f_impl(x):
  """
  Define the mapping function from upstream layer to the input of next layer
  this is just a default function that is linked to DEFAULT_MAP_F.
  (only implement things once if poissible)
  """
  return x


class SupervisedAutoencoder(Autoencoder, StackableNetwork):

  DEFAULT_MAP_F = map_f_impl

  supervision: nn.Sequential = None

  def __init__(self, color_channels):
    super().__init__(color_channels=color_channels)

    fc_layer_size = 16*8*8*color_channels
    self.supervision = nn.Sequential(
      FCView(),
      nn.Linear(in_features=fc_layer_size, out_features=100),
      nn.Linear(in_features=100, out_features=10),
    )

  def forward(self, x):
    encoding = self.encoder(x)
    prediction = self.supervision(encoding)
    decoding = self.decoder(encoding)
    return decoding, prediction


class NetworkStack(nn.Module):

  networks: List[StackableNetwork] = None
  map_to_input: Callable[[Tensor], Tensor] 
  
  def __init__(self,
    networks: List[Tuple[StackableNetwork, Callable[[Tensor], Tensor]]],
    train_every_pass=False
  ):
    super().__init__()
    self.networks = networks
  
  def forward(self, x):
    for i in range(len(self.networks) - 1):
      with no_grad():
        net, map_f = self.networks[i]
        x = net.calculate_upstream(x)
        x = map_f(x)
    decoding, prediction = self.networks[-1].forward(x)
    return decoding, prediction
    