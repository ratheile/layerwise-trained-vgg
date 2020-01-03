# adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
import torch
from torch import nn, Tensor
from typing import Tuple, List

cfgs = {
  'A': [ 64,  64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512],
  'B': [ 32,  32, 'M',  64,  64, 'M', 128, 128, 'M', 256, 256],
  'C': [ 16,  16, 'M',  32,  32, 'M',  64,  64, 'M', 128, 128],
  'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 1024, 1024]
}

class VGG(nn.Module):
  def __init__(self, 
    num_classes: int,
    dropout: float,
    img_size: int,
    vgg_version: str,
    init_weights=True):

    super().__init__()
    layers, trainable_modules = self.make_modules(cfgs[vgg_version], img_size)
    self.layers = nn.Sequential(*layers)
    self.trainable_modules = trainable_modules

    self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
    self.classifier = nn.Sequential(
      # take size from last vgg layer
      nn.Linear(cfgs[vgg_version][-1] * 4 * 4, 1024), 
      nn.LeakyReLU(inplace=True),
      nn.Dropout(dropout),
      nn.Linear(1024, num_classes),
    )

    self.vgg_dropout = dropout

    if init_weights:
        self._initialize_weights()

  def forward(self, x):
    with torch.no_grad():
      x = self.layers(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)
 
  def get_trainable_modules(self) -> List[Tuple[
    List[nn.Module],
    Tuple[int, int],
    int,
    List[nn.Module]
  ]]:
    return self.trainable_modules

  def make_modules(self, cfg, img_size, batch_norm=False) -> Tuple[
    List[nn.Module],
    List[Tuple[
      List[nn.Module],
      Tuple[int, int],
      int,
      List[nn.Module]
    ]]
  ]:

    all_layers = []
    trainable_layers = [] # only add trainable blocks to this
    trainable_layers_cs = [] # channel sizes
    trainable_layers_is = [] # image sizes
    maps = [] # these layers are untrainable maps

    in_channels = 3  # number of input channels

    mp =  (list(map(lambda x: x if x == 'M' else None, cfg)) + [None])[1:]
    cfg = list(filter(lambda t: t[0] != 'M',zip(cfg, mp)))
    
    for (out_channels, maxpl) in cfg:

      conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
      if batch_norm:
        module = [conv2d,
          nn.BatchNorm2d(out_channels),
          nn.LeakyReLU(inplace=True),
        ]
      else:
        module = [conv2d, nn.LeakyReLU(inplace=True)]

      
      all_layers += module
      trainable_layers.append(module) 
      trainable_layers_cs += [(in_channels, out_channels)]
      trainable_layers_is += [img_size]
      in_channels = out_channels

      if maxpl == 'M':
        maxpool = [nn.MaxPool2d(kernel_size=2, stride=2)]
        img_size = int(img_size / 2)
        all_layers += maxpool
        maps += maxpool
      else:
        maps += [None]

    # we split up these layers to map it to our stacked network
    # architecture:
    # modules are used directly in the sidecar autoencoder
    # max-pool layers become upstream maps, as they perfectly fit to this
    # idea of having a mapping in between modules that are trained
    # "sideways"
    return all_layers, list(zip(
      trainable_layers,
      trainable_layers_cs,
      trainable_layers_is, 
      maps))
