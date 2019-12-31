import torch
from torch import nn, Tensor
from torch import no_grad

from .stackable_network import StackableNetwork

from typing import List, Callable, Tuple

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