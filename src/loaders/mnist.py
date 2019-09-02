from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import os

class MnistLoader(DataLoader):

  batch_size = 128

  def __init__(self, path, download=False):

    img_transform = transforms.Compose([
      transforms.ToTensor(),
      # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      transforms.Normalize([0.5], [0.5])
    ])

    dataset = MNIST(path,
      transform=img_transform,
      download=download
    )

    super().__init__(
      dataset,
      batch_size=self.batch_size,
      shuffle=True
    )





