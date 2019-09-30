from torch import nn
from .interpolate import Interpolate
from .fcview import FCView

class OriginalAutoencoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential(
      nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
      nn.ReLU(True),
      nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
      nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
      nn.ReLU(True),
      nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
    )
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
      nn.ReLU(True),
      nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
      nn.ReLU(True),
      nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
      nn.Tanh()
    )

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

class Autoencoder(nn.Module):
  def __init__(self, color_channels=1):
    super().__init__()

    # Conv2d:      b,1,28,28   -->  b,8,28,28
    # MaxPool2d:   b,8,28,28   -->  b,8,14,14
    # Conv2d:      b,8,14,14   -->  b,16,14,14
    # MaxPool2d:   b,16,14,14  -->  b,16,7,7
    self.encoder = nn.Sequential(
      nn.Conv2d(in_channels=color_channels, out_channels=color_channels*8, kernel_size=3, stride=1, padding=1),   
      nn.BatchNorm2d(num_features=color_channels*8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
      nn.ReLU(True),
      nn.MaxPool2d(kernel_size=2),  
      nn.Conv2d(in_channels=color_channels*8, out_channels=color_channels*16, kernel_size=3, stride=1, padding=1), 
      nn.BatchNorm2d(num_features=color_channels*16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
      nn.ReLU(True),
      nn.MaxPool2d(kernel_size=2)   
    )

    # Interpolate:  b,16,7,7    -->  b,16,14,14
    # Conv2d:       b,16,14,14  -->  b,8,14,14
    # Interpolate:  b,8,14,14   -->  b,8,28,28
    # Conv2d:       b,16,14,14  -->  b,8,14,14
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

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x
  

class SupervisedAutoencoder(Autoencoder):
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
