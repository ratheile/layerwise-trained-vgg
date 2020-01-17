r"""
interpolate.py
==============

.. autosummary::
  modules.Interpolate
"""
from torch import nn

class Interpolate(nn.Module):
    r"""
    Interpolate function used for inverse convolution in
    Decoders
    """
    def __init__(self, scale_factor=2, mode="nearest"):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x