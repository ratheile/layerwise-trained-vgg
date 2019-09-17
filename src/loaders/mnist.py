from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from enum import Enum


class SetType(Enum):
  TRAIN = 'train'
  TEST = 'test'
  VALIDATION = 'validation'

class MnistLoader(DataLoader):

  batch_size = 128

  def __init__(self, path, download=False, 
    type:SetType = SetType.TRAIN,
    batch_size=128):

    self.batch_size = batch_size

    img_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([0.5], [0.5])
    ])

    train = (type == SetType.TRAIN)

    dataset = MNIST(
      root=path,
      transform=img_transform,
      train=train,
      download=download
    )

    super().__init__(
      dataset,
      batch_size=self.batch_size,
      shuffle=True
    )





