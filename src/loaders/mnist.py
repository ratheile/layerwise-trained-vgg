from torch import Tensor, randperm
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms
import os
from enum import Enum
from PIL import Image
from typing import List, Tuple

import numpy as np

class MnistSubsetSampler(Sampler):
  def __init__(self, indices):
      self.indices = indices

  def __iter__(self):
      return (self.indices[i] for i in randperm(len(self.indices)))

  def __len__(self):
      return len(self.indices)

def semi_supervised_mnist(
  root,
  supervised_ratio=0.1,
  batch_size=128,
  download=False
) -> Tuple[DataLoader, DataLoader, DataLoader]:    


  img_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([0.5], [0.5])
    ])

  mnist_train = MNIST(
      root=root,
      transform=img_transform,
      train=True,
      download=download
    )

  mnist_test = MNIST(
      root=root,
      transform=img_transform,
      train=False,
      download=download
    )

  mnist_size = len(mnist_train)
  assert supervised_ratio < 1 and supervised_ratio > 0
  ss_ds_len = int(mnist_size * supervised_ratio)

  index_total = np.arange(mnist_size)
  np.random.shuffle(index_total)

  supervised_sampler = MnistSubsetSampler(index_total[:ss_ds_len])
  unsupvised_sampler = MnistSubsetSampler(index_total[ss_ds_len:])

  unsupvised_loader = DataLoader(mnist_train, 
    sampler=unsupvised_sampler,
    batch_size=int(batch_size - (batch_size * supervised_ratio))
  )

  supervised_loader = DataLoader(mnist_train, 
    sampler=supervised_sampler,
    batch_size=int(batch_size * supervised_ratio)
  )

  test_loader = DataLoader(mnist_test, 
    batch_size=batch_size
  )

  return supervised_loader, unsupvised_loader, test_loader