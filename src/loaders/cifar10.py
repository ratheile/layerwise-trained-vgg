from torch import Tensor, randperm
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms
import os
from enum import Enum
from PIL import Image
from typing import List, Tuple
import logging

import numpy as np

class CifarSubsetSampler(Sampler):
  def __init__(self, indices):
      self.indices = indices

  def __iter__(self):
      return (self.indices[i] for i in randperm(len(self.indices)))

  def __len__(self):
      return len(self.indices)

def semi_supervised_cifar10(
  root,
  supervised_ratio=0.1,
  batch_size=128,
  download=False
) -> Tuple[DataLoader, DataLoader, DataLoader]:    


  img_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([0.5], [0.5])
    ])

  ds_train = CIFAR10(
      root=root,
      transform=img_transform,
      train=True,
      download=download
    )

  ds_test = CIFAR10(
      root=root,
      transform=img_transform,
      train=False,
      download=download
    )

  ds_size = len(ds_train)
  assert supervised_ratio < 1 and supervised_ratio > 0
  ss_ds_len = int(ds_size * supervised_ratio)

  index_total = np.arange(ds_size)
  np.random.shuffle(index_total)

  supervised_sampler = CifarSubsetSampler(index_total[:ss_ds_len])
  unsupervised_sampler = CifarSubsetSampler(index_total[ss_ds_len:])

  us_bs = int(batch_size - (batch_size * supervised_ratio))
  s_bs = int(batch_size * supervised_ratio)
  
  assert us_bs + s_bs == batch_size

  logging.info(f"Dataset (supervised={ss_ds_len}/total={ds_size}) us_bs={us_bs} s_bs={s_bs}")

  unsupervised_loader = DataLoader(ds_train, 
    sampler=unsupervised_sampler,
    batch_size=us_bs
  )

  supervised_loader = DataLoader(ds_train, 
    sampler=supervised_sampler,
    batch_size=s_bs
  )

  test_loader = DataLoader(ds_test, 
    batch_size=batch_size
  )

  return supervised_loader, unsupervised_loader, test_loader
