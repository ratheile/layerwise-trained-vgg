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
  download=False,
  num_workers=(6,6,2)
) -> Tuple[DataLoader, DataLoader, DataLoader]:    

  
  img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
  ])

  # img_transform = transforms.Compose([
  #   transforms.RandomCrop(32, padding=4),
  #   transforms.RandomHorizontalFlip(),                                
  #   transforms.ToTensor(),
  #   transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
  # ])

  img_transform_s = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(45),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),                                
    transforms.ToTensor(),
    transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284)) # TODO: WHY???
  ])

  ds_train_us = CIFAR10(
      root=root,
      transform=img_transform_s,
      train=True,
      download=download
    )

  ds_train_s = CIFAR10(
      root=root,
      transform=img_transform_s,
      train=True,
      download=download
  )

  ds_test = CIFAR10(
      root=root,
      transform=img_transform_s,
      train=False,
      download=download
    )

  ds_size = len(ds_train_s)
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
  logging.info(f"Using workers (unsupervised|supervised|test) {num_workers}")

  unsupervised_loader = DataLoader(ds_train_us, 
    num_workers=num_workers[0],
    sampler=unsupervised_sampler,
    batch_size=us_bs
  )

  supervised_loader = DataLoader(ds_train_s, 
    num_workers=num_workers[1],
    sampler=supervised_sampler,
    batch_size=s_bs
  )

  test_loader = DataLoader(ds_test, 
    num_workers=num_workers[2],
    batch_size=batch_size
  )

  return supervised_loader, unsupervised_loader, test_loader
